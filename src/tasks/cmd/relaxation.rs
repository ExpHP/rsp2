use super::trial::TrialDir;
use super::GammaSystemAnalysis;
use super::potential::{PotentialBuilder, DiffFn, DynFlatDiffFn};
use super::CliArgs;
use super::{write_eigen_info_for_humans, write_eigen_info_for_machines};
use super::SupercellSpecExt;
use super::carbon;

use ::{FailResult, FailOk};
use ::rsp2_tasks_config::{self as cfg, Settings};
use ::traits::{AsPath};
use ::phonopy::{DirWithBands};

use ::math::basis::Basis3;

use ::rsp2_slice_math::{v, V, vdot};

use ::slice_of_array::prelude::*;
use ::rsp2_array_types::{V3};
use ::rsp2_structure::supercell::{self, SupercellToken};
use ::rsp2_structure::{ElementStructure};
use ::rsp2_structure::{CoordsKind};
use ::rsp2_structure_gen::Assemble;
use ::phonopy::Builder as PhonopyBuilder;
use ::math::bands::ScMatrix;

impl TrialDir {
    /// NOTE: This writes to fixed filepaths in the trial directory
    ///       and is not designed to be called multiple times.
    pub(crate) fn do_main_ev_loop(
        &self,
        settings: &Settings,
        cli: &CliArgs,
        pot: &PotentialBuilder,
        atom_layers: &Option<Vec<usize>>,
        layer_sc_mats: &Option<Vec<ScMatrix>>,
        phonopy: &PhonopyBuilder,
        original_structure: ElementStructure,
    ) -> FailResult<(ElementStructure, GammaSystemAnalysis, DirWithBands<Box<AsPath>>)>
    {
        let mut from_structure = original_structure;
        let mut loop_state = EvLoopFsm::new(&settings.ev_loop);
        loop {
            // move out of from_structure so that Rust's control-flow analysis
            // will make sure we put something back.
            let structure = from_structure;
            let iteration = loop_state.iteration;

            trace!("============================");
            trace!("Begin relaxation # {}", iteration);

            let structure = do_relax(pot, &settings.cg, &settings.potential, structure)?;

            trace!("============================");

            self.write_poscar(
                &format!("structure-{:02}.1.vasp", iteration),
                &format!("Structure after CG round {}", iteration),
                &structure,
            )?;

            let aux_info = {
                use super::ev_analyses::*;

                // HACK
                let masses = {
                    structure.metadata().iter()
                        .map(|&s| ::common::element_mass(s))
                        .collect()
                };

                super::aux_info::Info {
                    atom_layers:   atom_layers.clone().map(AtomLayers),
                    layer_sc_mats: layer_sc_mats.clone().map(LayerScMatrices),
                    atom_masses:   Some(AtomMasses(masses)),
                }
            };

            let (bands_dir, evals, evecs, ev_analysis) = {
                self.save_analysis_aux_info(&aux_info)?;

                let save_bands = match cli.save_bands {
                    true => Some(self.save_bands_dir()),
                    false => None,
                };

                self.do_post_relaxation_computations(
                    settings, save_bands.as_ref(), pot, aux_info, phonopy, &structure,
                )?
            };

            {
                let file = self.create_file(format!("eigenvalues.{:02}", iteration))?;
                write_eigen_info_for_machines(&ev_analysis, file)?;
                write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;
            }

            let (structure, did_chasing) = self.maybe_do_ev_chasing(
                settings, pot, structure, &ev_analysis, &evals, &evecs,
            )?;

            self.write_poscar(
                &format!("structure-{:02}.2.vasp", iteration),
                &format!("Structure after eigenmode-chasing round {}", iteration),
                &structure,
            )?;

            warn_on_improvable_lattice_params(pot, &structure)?;

            match loop_state.step(did_chasing) {
                EvLoopStatus::KeepGoing => {
                    from_structure = structure;
                    continue;
                },
                EvLoopStatus::Done => {
                    return Ok((structure, ev_analysis, bands_dir));
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
        structure: ElementStructure,
        ev_analysis: &GammaSystemAnalysis,
        evals: &[f64],
        evecs: &Basis3,
    ) -> FailResult<(ElementStructure, DidEvChasing)>
    {Ok({
        use super::acoustic_search::ModeKind;
        let structure = structure;
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
            0 => (structure, DidEvChasing(false)),
            n => {
                trace!("Chasing {} bad eigenvectors...", n);
                let structure = do_eigenvector_chase(
                    pot,
                    &settings.ev_chase,
                    &settings.potential,
                    structure,
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
    potential_settings: &cfg::Potential,
    structure: ElementStructure,
) -> FailResult<ElementStructure>
{Ok({
    let sc_dims = potential_settings.supercell.dim_for_unitcell(structure.lattice());
    let (supercell, sc_token) = supercell::diagonal(sc_dims).build(structure);

    let mut flat_diff_fn = pot.threaded(true).initialize_flat_diff_fn(supercell.clone())?;
    let relaxed_flat = ::rsp2_minimize::acgsd(
        cg_settings,
        supercell.to_carts().flat(),
        &mut *flat_diff_fn,
    ).unwrap().position;

    let supercell = supercell.with_coords(CoordsKind::Carts(relaxed_flat.nest().to_vec()));
    multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?
})}

fn do_eigenvector_chase(
    pot: &PotentialBuilder,
    chase_settings: &cfg::EigenvectorChase,
    potential_settings: &cfg::Potential,
    mut structure: ElementStructure,
    bad_evecs: &[(String, &[V3])],
) -> FailResult<ElementStructure>
{Ok({
    match *chase_settings {
        cfg::EigenvectorChase::OneByOne => {
            for &(ref name, evec) in bad_evecs {
                let (alpha, new_structure) = do_minimize_along_evec(pot, potential_settings, structure, &evec[..])?;
                info!("Optimized along {}, a = {:e}", name, alpha);

                structure = new_structure;
            }
            structure
        },
        cfg::EigenvectorChase::Acgsd(ref cg_settings) => {
            let evecs: Vec<_> = bad_evecs.iter().map(|&(_, ev)| ev).collect();
            do_cg_along_evecs(
                pot,
                cg_settings,
                potential_settings,
                structure,
                &evecs[..],
            )?
        },
    }
})}

fn do_cg_along_evecs<V, I>(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Acgsd,
    potential_settings: &cfg::Potential,
    structure: ElementStructure,
    evecs: I,
) -> FailResult<ElementStructure>
where
    V: AsRef<[V3]>,
    I: IntoIterator<Item=V>,
{Ok({
    let evecs: Vec<_> = evecs.into_iter().collect();
    let refs: Vec<_> = evecs.iter().map(|x| x.as_ref()).collect();
    _do_cg_along_evecs(pot, cg_settings, potential_settings, structure, &refs)?
})}

fn _do_cg_along_evecs(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Acgsd,
    potential_settings: &cfg::Potential,
    structure: ElementStructure,
    evecs: &[&[V3]],
) -> FailResult<ElementStructure>
{Ok({
    let sc_dims = potential_settings.supercell.dim_for_unitcell(structure.lattice());
    let (mut supercell, sc_token) = supercell::diagonal(sc_dims).build(structure);
    let evecs: Vec<_> = evecs.iter().map(|ev| sc_token.replicate(ev)).collect();

    let flat_evecs: Vec<_> = evecs.iter().map(|ev| ev.flat()).collect();
    let init_pos = supercell.to_carts();

    let mut flat_diff_fn = pot.threaded(true).initialize_flat_diff_fn(supercell.clone())?;
    let relaxed_coeffs = ::rsp2_minimize::acgsd(
        cg_settings,
        &vec![0.0; evecs.len()],
        &mut *lammps_constrained_diff_fn(&mut *flat_diff_fn, init_pos.flat(), &flat_evecs),
    ).unwrap().position;

    let final_flat_pos = flat_constrained_position(init_pos.flat(), &relaxed_coeffs, &flat_evecs);
    supercell.carts_mut().copy_from_slice(final_flat_pos.nest());
    multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, supercell)?
})}

fn do_minimize_along_evec(
    pot: &PotentialBuilder,
    settings: &cfg::Potential,
    structure: ElementStructure,
    evec: &[V3],
) -> FailResult<(f64, ElementStructure)>
{Ok({
    let sc_dims = settings.supercell.dim_for_unitcell(structure.lattice());
    let (structure, sc_token) = supercell::diagonal(sc_dims).build(structure);
    let evec = sc_token.replicate(evec);
    let mut diff_fn = pot.threaded(true).initialize_flat_diff_fn(structure.clone())?;

    let from_structure = structure;
    let direction = &evec[..];
    let from_pos = from_structure.to_carts();
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
    let structure = from_structure.with_coords(CoordsKind::Carts(pos.nest().to_vec()));

    (alpha, multi_threshold_deconstruct(sc_token, 1e-10, 1e-3, structure)?)
})}

fn warn_on_improvable_lattice_params(
    pot: &PotentialBuilder,
    structure: &ElementStructure,
) -> FailResult<()>
{Ok({
    const SCALE_AMT: f64 = 1e-6;
    let mut diff_fn = pot.initialize_diff_fn(structure.clone())?;
    let center_value = diff_fn.compute_value(structure)?;

    let shrink_value = {
        let mut structure = structure.clone();
        structure.scale_vecs(&[1.0 - SCALE_AMT, 1.0 - SCALE_AMT, 1.0]);
        diff_fn.compute_value(&structure)?
    };

    let enlarge_value = {
        let mut structure = structure.clone();
        structure.scale_vecs(&[1.0 + SCALE_AMT, 1.0 + SCALE_AMT, 1.0]);
        diff_fn.compute_value(&structure)?
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
fn lammps_constrained_diff_fn<'a>(
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

fn multi_threshold_deconstruct(
    sc_token: SupercellToken,
    warn: f64,
    fail: f64,
    supercell: ElementStructure,
) -> FailResult<ElementStructure>
{Ok({
    match sc_token.deconstruct(warn, supercell.clone()) {
        Ok(x) => x,
        Err(e) => {
            warn!("{}", e);
            sc_token.deconstruct(fail, supercell)?
        }
    }
})}

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

pub(crate) fn optimize_layer_parameters(
    settings: &cfg::ScaleRanges,
    pot: &PotentialBuilder,
    mut layer_builder: Assemble,
) -> FailResult<Assemble>
{Ok({
    pub use ::rsp2_minimize::exact_ls::{Value, Golden};
    use ::rsp2_tasks_config::{ScaleRanges, ScaleRange, ScaleRangesLayerSepStyle};
    use ::std::cell::RefCell;

    let ScaleRanges {
        parameter: ref parameter_spec,
        layer_sep: ref layer_sep_spec,
        warn: warn_threshold,
        layer_sep_style,
        repeat_count,
    } = *settings;

    let n_seps = layer_builder.layer_seps().len();

    // abuse RefCell for some DRYness
    let layer_builder = RefCell::new(layer_builder);
    {
        let layer_builder = &layer_builder;

        // Make a bunch of closures representing quantities that can be optimized.
        // (NOTE: this is the only part that really contains stuff specific to layers)
        let optimizables = {
            let mut optimizables: Vec<(_, Box<Fn(f64)>)> = vec![];
            optimizables.push((
                (format!("lattice parameter"), parameter_spec.clone()),
                Box::new(|s| {
                    layer_builder.borrow_mut().scale = s;
                }),
            ));

            match layer_sep_style {
                ScaleRangesLayerSepStyle::Individual => {
                    for i in 0..n_seps {
                        optimizables.push((
                            (format!("layer sep {}", i), layer_sep_spec.clone()),
                            Box::new(move |s| {
                                layer_builder.borrow_mut().layer_seps()[i] = s;
                            }),
                        ));
                    }
                },
                ScaleRangesLayerSepStyle::Uniform => {
                    optimizables.push((
                        (format!("layer sep"), layer_sep_spec.clone()),
                        Box::new(move |s| {
                            for dest in layer_builder.borrow_mut().layer_seps() {
                                *dest = s;
                            }
                        })
                    ))
                },
            }
            optimizables
        };

        // Set reasonable values before initializing LAMMPS
        //
        // exact specs: set the value
        // range specs: start with reasonable defaults ('guess' in config)
        for &((_, ref spec), ref setter) in &optimizables {
            match *spec {
                ScaleRange::Exact(value) |
                ScaleRange::Range { range: _, guess: Some(value) } => {
                    setter(value);
                },
                ScaleRange::Range { range: _, guess: None } => {},
            }
        }

        let get_value = || FailOk({
            // FIXME: I'm not sure why this is a one-off computation; I should try
            //        reusing a single lammps instance and see what happens.
            //
            //        (maybe I had issues with "lost atoms" errors or something)
            pot.one_off().compute_value(&carbon(layer_builder.borrow().assemble()))?
        });

        // optimize them one-by-one.
        //
        // Repeat the whole process repeat_count times.
        // In future iterations, parameters other than the one currently being
        // relaxed may be set to different, better values, which may in turn
        // cause different values to be chosen for the earlier parameters.
        for _ in 0..repeat_count {
            for &((ref name, ref spec), ref setter) in &optimizables {
                trace!("Optimizing {}", name);

                let best = match *spec {
                    ScaleRange::Exact(value) => value,
                    ScaleRange::Range { guess: _, range } => {
                        let best = Golden::new()
                            .stop_condition(&from_json!({"interval-size": 1e-7}))
                            .run(range, |a| {
                                setter(a);
                                get_value().map(Value)
                            })??; // ?!??!!!?

                        if let Some(thresh) = warn_threshold {
                            // use signed differences so that all values outside violate the threshold
                            let lo = range.0.min(range.1);
                            let hi = range.0.max(range.1);
                            if (best - range.0).min(range.1 - best) / (range.1 - range.0) < thresh {
                                warn!("Relaxed value of '{}' is suspiciously close to limits!", name);
                                warn!("  lo: {:e}", lo);
                                warn!(" val: {:e}", best);
                                warn!("  hi: {:e}", hi);
                            }
                        }

                        info!("Optimized {}: {} (from {:?})", name, best, range);
                        best
                    },
                }; // let best = { ... }

                setter(best);
            } // for ... in optimizables
        } // for ... in repeat_count
    } // RefCell borrow scope

    layer_builder.into_inner()
})}

//-----------------------------------------------------------------------------
