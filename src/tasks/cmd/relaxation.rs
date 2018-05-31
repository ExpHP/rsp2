use super::trial::TrialDir;
use super::GammaSystemAnalysis;
use super::potential::{PotentialBuilder, DiffFn, DynFlatDiffFn};
use super::{write_eigen_info_for_humans, write_eigen_info_for_machines};

use ::{FailResult, FailOk};
use ::rsp2_tasks_config::{self as cfg, Settings};
use ::cmd::stored_structure::StoredStructure;
use ::meta::prelude::*;
use ::meta::{Mass, Element, Layer};
use ::hlist_aliases::*;

use ::math::basis::Basis3;
use ::math::bands::ScMatrix;
use ::phonopy::Builder as PhonopyBuilder;
use ::util::ext_traits::OptionResultExt;

use ::slice_of_array::prelude::*;
use ::rsp2_slice_math::{v, V, vdot};
use ::rsp2_array_types::{V3};
use ::rsp2_structure::{Coords};
use ::std::rc::Rc;
use math::bonds::FracBonds;

pub trait EvLoopDiagonalizer {
    type ExtraOut;

    fn do_post_relaxation_computations(
        &self,
        settings: &cfg::Settings,
        pot: &PotentialBuilder,
        phonopy: &PhonopyBuilder,
        stored: &StoredStructure,
    ) -> FailResult<(Vec<f64>, Basis3, Self::ExtraOut)>;
}


impl TrialDir {
    /// NOTE: This writes to fixed filepaths in the trial directory
    ///       and is not designed to be called multiple times.
    pub(crate) fn do_main_ev_loop<ExtraOut>(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        layer_sc_mats: &Option<Vec<ScMatrix>>,
        phonopy: &PhonopyBuilder,
        diagonalizer: impl EvLoopDiagonalizer<ExtraOut=ExtraOut>,
        original_coords: Coords,
        meta: HList3<
            Rc<[Element]>,
            Rc<[Mass]>,
            Option<Rc<[Layer]>>,
        >,
    ) -> FailResult<(Coords, GammaSystemAnalysis, ExtraOut)>
    {
        let mut from_coords = original_coords;
        let mut loop_state = EvLoopFsm::new(&settings.ev_loop);
        loop {
            // move out of from_coords so that Rust's control-flow analysis
            // will make sure we put something back.
            let coords = from_coords;
            let iteration = loop_state.iteration;

            trace!("============================");
            trace!("Begin relaxation # {}", iteration);

            let coords = do_relax(pot, &settings.cg, coords, meta.sift())?;

            trace!("============================");

            let (evals, evecs, extra_out) = {
                let subdir = format!("ev-loop-{:02}.1.structure", iteration);
                self.write_stored_structure(
                    &subdir,
                    &format!("Structure after CG round {}", iteration),
                    &coords, meta.sift(), layer_sc_mats,
                )?;

                let stored = self.read_stored_structure(&subdir)?;

                diagonalizer.do_post_relaxation_computations(settings, pot, phonopy, &stored)?
            };

            trace!("============================");
            trace!("Finished diagonalization");

            // FIXME: only the CartBonds need to be recomputed each iteration;
            //       we could keep the FracBonds around between iterations.
            let bonds = settings.bond_radius.map(|bond_radius| FailOk({
                trace!("Computing bonds");
                FracBonds::from_brute_force_very_dumb(&coords, bond_radius)?
            })).fold_ok()?;

            trace!("Computing eigensystem info");

            let ev_analysis = super::run_gamma_system_analysis(
                &settings, pot, &coords, meta.sift(), &layer_sc_mats, &evals, &evecs, &bonds,
            )?;

            {
                let file = self.create_file(format!("eigenvalues.{:02}", iteration))?;
                write_eigen_info_for_machines(&ev_analysis, file)?;
                write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;
            }

            let (coords, did_chasing) = self.maybe_do_ev_chasing(
                settings, pot, coords, meta.sift(), &ev_analysis, &evals, &evecs,
            )?;

            self.write_stored_structure(
                &format!("ev-loop-{:02}.2.structure", iteration),
                &format!("Structure after eigenmode-chasing round {}", iteration),
                &coords, meta.sift(), layer_sc_mats,
            )?;

            warn_on_improvable_lattice_params(pot, &coords, meta.sift())?;

            match loop_state.step(did_chasing) {
                EvLoopStatus::KeepGoing => {
                    from_coords = coords;
                    continue;
                },
                EvLoopStatus::Done => {
                    return Ok((coords, ev_analysis, extra_out));
                },
                EvLoopStatus::ItsBadGuys(msg) => {
                    bail!("{}", msg);
                },
            }
            // unreachable
        }
    }

    fn maybe_do_ev_chasing(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        coords: Coords,
        meta: HList2<
            Rc<[Element]>,
            Rc<[Mass]>,
        >,
        ev_analysis: &GammaSystemAnalysis,
        evals: &[f64],
        evecs: &Basis3,
    ) -> FailResult<(Coords, DidEvChasing)>
    {Ok({
        use super::acoustic_search::ModeKind;
        let bad_evs: Vec<_> = {
            let classifications = ev_analysis.ev_classifications.as_ref().expect("(bug) always computed!");
            izip!(1.., evals, &evecs.0, &classifications.0)
                .filter(|&(_, _, _, kind)| kind == &ModeKind::Imaginary)
                .map(|(i, freq, evec, _)| {
                    let name = format!("band {} ({})", i, freq);
                    (name, evec.as_real_checked())
                }).collect()
        };

        match bad_evs.len() {
            0 => (coords, DidEvChasing(false)),
            n => {
                trace!("Chasing {} bad eigenvectors...", n);
                let structure = do_eigenvector_chase(
                    pot,
                    &settings.ev_chase,
                    coords,
                    meta.sift(),
                    &bad_evs[..],
                )?;
                (structure, DidEvChasing(true))
            }
        }
    })}
}

struct EvLoopFsm {
    config: cfg::EvLoop,
    iteration: u32,
    all_ok_count: u32,
}

pub enum EvLoopStatus {
    KeepGoing,
    Done,
    ItsBadGuys(&'static str),
}

// just a named bool for documentation
pub struct DidEvChasing(bool);

impl EvLoopFsm {
    pub fn new(config: &cfg::EvLoop) -> Self
    { EvLoopFsm {
        config: config.clone(),
        iteration: 1,
        all_ok_count: 0,
    }}

    pub fn step(&mut self, did: DidEvChasing) -> EvLoopStatus {
        self.iteration += 1;
        match did {
            DidEvChasing(true) => {
                self.all_ok_count = 0;
                if self.iteration > self.config.max_iter {
                    if self.config.fail {
                        EvLoopStatus::ItsBadGuys("Too many relaxation steps!")
                    } else {
                        warn!("Too many relaxation steps!");
                        EvLoopStatus::Done
                    }
                } else {
                    EvLoopStatus::KeepGoing
                }
            },
            DidEvChasing(false) => {
                self.all_ok_count += 1;
                if self.all_ok_count >= self.config.min_positive_iter {
                    EvLoopStatus::Done
                } else {
                    EvLoopStatus::KeepGoing
                }
            },
        }
    }
}

//-----------------------------------------------------------------------------

fn do_relax(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Acgsd,
    coords: Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
) -> FailResult<Coords>
{Ok({
    let mut flat_diff_fn = pot.threaded(true).initialize_flat_diff_fn(&coords, meta.sift())?;
    let relaxed_flat = ::rsp2_minimize::acgsd(
        cg_settings,
        coords.to_carts().flat(),
        &mut *flat_diff_fn,
    ).unwrap().position;
    coords.with_carts(relaxed_flat.nest().to_vec())
})}

fn do_eigenvector_chase(
    pot: &PotentialBuilder,
    chase_settings: &cfg::EigenvectorChase,
    mut coords: Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
    bad_evecs: &[(String, &[V3])],
) -> FailResult<Coords>
{Ok({
    match chase_settings {
        cfg::EigenvectorChase::OneByOne => {
            for (name, evec) in bad_evecs {
                let (alpha, new_coords) = do_minimize_along_evec(pot, coords, meta.sift(), &evec[..])?;
                info!("Optimized along {}, a = {:e}", name, alpha);

                coords = new_coords;
            }
            coords
        },
        cfg::EigenvectorChase::Acgsd(cg_settings) => {
            let evecs: Vec<_> = bad_evecs.iter().map(|&(_, ev)| ev).collect();
            do_cg_along_evecs(
                pot,
                cg_settings,
                coords,
                meta.sift(),
                &evecs[..],
            )?
        },
    }
})}

fn do_cg_along_evecs<V, I>(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Acgsd,
    coords: Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
    evecs: I,
) -> FailResult<Coords>
where
    V: AsRef<[V3]>,
    I: IntoIterator<Item=V>,
{Ok({
    let evecs: Vec<_> = evecs.into_iter().collect();
    let refs: Vec<_> = evecs.iter().map(|x| x.as_ref()).collect();
    _do_cg_along_evecs(pot, cg_settings, coords, meta, &refs)?
})}

fn _do_cg_along_evecs(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Acgsd,
    coords: Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
    evecs: &[&[V3]],
) -> FailResult<Coords>
{Ok({
    let flat_evecs: Vec<_> = evecs.iter().map(|ev| ev.flat()).collect();
    let init_pos = coords.to_carts();

    let mut flat_diff_fn = pot.threaded(true).initialize_flat_diff_fn(&coords, meta.sift())?;
    let relaxed_coeffs = ::rsp2_minimize::acgsd(
        cg_settings,
        &vec![0.0; evecs.len()],
        &mut *constrained_diff_fn(&mut *flat_diff_fn, init_pos.flat(), &flat_evecs),
    ).unwrap().position;

    let final_flat_pos = flat_constrained_position(init_pos.flat(), &relaxed_coeffs, &flat_evecs);
    coords.with_carts(final_flat_pos.nest().to_vec())
})}

fn do_minimize_along_evec(
    pot: &PotentialBuilder,
    from_coords: Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
    evec: &[V3],
) -> FailResult<(f64, Coords)>
{Ok({
    let mut diff_fn = pot.threaded(true).initialize_flat_diff_fn(&from_coords, meta.sift())?;

    let direction = &evec[..];
    let from_pos = from_coords.to_carts();
    let pos_at_alpha = |alpha| {
        let V(pos) = v(from_pos.flat()) + alpha * v(direction.flat());
        pos
    };
    let alpha = ::rsp2_minimize::exact_ls(0.0, 1e-4, |alpha| {
        let gradient = diff_fn(&pos_at_alpha(alpha))?.1;
        let slope = vdot(&gradient[..], direction.flat());
        FailOk(::rsp2_minimize::exact_ls::Slope(slope))
    })??.alpha;
    let pos = pos_at_alpha(alpha);

    (alpha, from_coords.with_carts(pos.nest().to_vec()))
})}

fn warn_on_improvable_lattice_params(
    pot: &PotentialBuilder,
    coords: &Coords,
    meta: HList2<
        Rc<[Element]>,
        Rc<[Mass]>,
    >,
) -> FailResult<()>
{Ok({
    const SCALE_AMT: f64 = 1e-6;
    let mut diff_fn = pot.initialize_diff_fn(coords, meta.sift())?;
    let center_value = diff_fn.compute_value(coords, meta.sift())?;

    let shrink_value = {
        let mut coords = coords.clone();
        coords.scale_vecs(&[1.0 - SCALE_AMT, 1.0 - SCALE_AMT, 1.0]);
        diff_fn.compute_value(&coords, meta.sift())?
    };

    let enlarge_value = {
        let mut coords = coords.clone();
        coords.scale_vecs(&[1.0 + SCALE_AMT, 1.0 + SCALE_AMT, 1.0]);
        diff_fn.compute_value(&coords, meta.sift())?
    };

    if shrink_value.min(enlarge_value) < center_value {
        warn!("Better value found at nearby lattice parameter:");
        warn!(" Smaller: {}", shrink_value);
        warn!(" Current: {}", center_value);
        warn!("  Larger: {}", enlarge_value);
    }
})}

fn flat_constrained_position(
    flat_init_pos: &[f64],
    coeffs: &[f64],
    flat_evecs: &[&[f64]],
) -> Vec<f64>
{
    let flat_d_pos = dot_vec_mat_dumb(coeffs, flat_evecs);
    let V(flat_pos): V<Vec<_>> = v(flat_init_pos) + v(flat_d_pos);
    flat_pos
}

// cg differential function along a restricted set of eigenvectors.
//
// There will be one coordinate for each eigenvector.
fn constrained_diff_fn<'a>(
    // operates on 3N coords
    flat_3n_diff_fn: &'a mut DynFlatDiffFn<'a>,
    // K values, K <= 3N
    flat_init_pos: &'a [f64],
    // K eigenvectors
    flat_evs: &'a [&[f64]],
) -> Box<FnMut(&[f64]) -> FailResult<(f64, Vec<f64>)> + 'a>
{
    Box::new(move |coeffs| Ok({
        assert_eq!(coeffs.len(), flat_evs.len());

        // This is dead simple.
        // The kth element of the new gradient is the slope along the kth ev.
        // The change in position is a sum over contributions from each ev.
        // These relationships have a simple expression in terms of
        //   the matrix whose columns are the selected eigenvectors.
        // (though the following is transposed for our row-centric formalism)
        let flat_pos = flat_constrained_position(flat_init_pos, coeffs, flat_evs);

        let (value, flat_grad) = flat_3n_diff_fn(&flat_pos)?;

        let grad = dot_mat_vec_dumb(flat_evs, &flat_grad);
        (value, grad)
    }))
}

//----------------------
// a slice of slices is a really dumb representation for a matrix
// but we do not require performance where this is used, so whatever

fn dot_vec_mat_dumb(vec: &[f64], mat: &[&[f64]]) -> Vec<f64>
{
    assert_eq!(mat.len(), vec.len());
    assert_ne!(mat.len(), 0, "cannot determine width of matrix with no rows");
    let init = v(vec![0.0; mat[0].len()]);
    let V(out) = izip!(mat, vec)
        .fold(init, |acc, (&row, &alpha)| acc + alpha * v(row));
    out
}

fn dot_mat_vec_dumb(mat: &[&[f64]], vec: &[f64]) -> Vec<f64>
{ mat.iter().map(|row| vdot(vec, row)).collect() }

//-----------------------------------------------------------------------------
