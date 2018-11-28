/* ********************************************************************** **
**  This file is part of rsp2.                                            **
**                                                                        **
**  rsp2 is free software: you can redistribute it and/or modify it under **
**  the terms of the GNU General Public License as published by the Free  **
**  Software Foundation, either version 3 of the License, or (at your     **
**  option) any later version.                                            **
**                                                                        **
**      http://www.gnu.org/licenses/                                      **
**                                                                        **
** Do note that, while the whole of rsp2 is licensed under the GPL, many  **
** parts of it are licensed under more permissive terms.                  **
** ********************************************************************** */

use crate::{FailResult, FailOk};
use crate::filetypes::stored_structure::StoredStructure;
use crate::potential::{self, PotentialBuilder, DiffFn, BondDiffFn, DynFlatDiffFn};
use crate::meta::{self, prelude::*};
use crate::hlist_aliases::*;
use crate::math::basis::{Basis3, EvDirection};

use super::trial::TrialDir;
use super::GammaSystemAnalysis;
use super::{write_eigen_info_for_humans, write_eigen_info_for_machines};
use super::{EvLoopStructureKind, Iteration};
use super::param_optimization::RelaxationOptimizationHelper;

use rsp2_tasks_config::{self as cfg, Settings};

use slice_of_array::prelude::*;
use rsp2_slice_math::{v, V, vdot};
use rsp2_array_types::{V3};
use rsp2_structure::{Coords};
use rsp2_minimize::{cg};

use std::rc::Rc;

pub trait EvLoopDiagonalizer {
    type ExtraOut;

    fn do_post_relaxation_computations(
        &self,
        // (even if no impl uses the TrialDir, it is passed around so that files can easily be
        //  written for debugging purposes)
        trial: &TrialDir,
        settings: &cfg::Settings,
        pot: &PotentialBuilder,
        stored: &StoredStructure,
        stop_after_dynmat: bool, // HACK
        // HACK: Used to set the filename for saving the gamma dynmat.
        //       That shouldn't even be a responsibility of the function, but IIRC it does this
        //       for easier debugging in case an error occurs shortly after computing the matrix
        iteration: Option<Iteration>, // HACK
    ) -> FailResult<(Vec<f64>, Basis3, Self::ExtraOut)>;

    fn allow_unfold_bands(&self) -> bool;
}

impl TrialDir {
    /// NOTE: This writes to fixed filepaths in the trial directory
    ///       and is not designed to be called multiple times.
    ///
    /// The `ExtraOut` returned is the one from the final iteration.
    pub(crate) fn do_main_ev_loop<ExtraOut>(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        diagonalizer: impl EvLoopDiagonalizer<ExtraOut=ExtraOut>,
        original_coords: Coords,
        meta: HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
        stop_after_dynmat: bool, // HACK
    ) -> FailResult<(Coords, GammaSystemAnalysis, ExtraOut)>
    {
        let mut from_coords = original_coords;
        let mut loop_state = EvLoopFsm::new(&settings.ev_loop);
        loop {
            // move out of from_coords so that Rust's control-flow analysis
            // will make sure we put something back.
            let coords = from_coords;
            let iteration = loop_state.iteration;

            let coords = self.do_ev_loop_stuff_before_diagonalization(
                &settings, pot, meta.sift(), iteration, coords,
            )?;

            let (evals, evecs, extra_out) = {
                let subdir = self.structure_path(EvLoopStructureKind::PreEvChase(iteration));
                let stored = self.read_stored_structure(&subdir)?;
                diagonalizer.do_post_relaxation_computations(
                    &self, settings, pot, &stored, stop_after_dynmat, Some(iteration),
                )?
            };

            trace!("============================");
            trace!("Finished diagonalization");

            let (ev_analysis, coords, did_chasing) = {
                self.do_ev_loop_stuff_after_diagonalization(
                    &settings, pot, meta.sift(), iteration, coords, &evals, &evecs,
                    diagonalizer.allow_unfold_bands(),
                )?
            };

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

    pub(in ::cmd) fn do_ev_loop_stuff_before_diagonalization(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        meta: HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
        iteration: Iteration,
        coords: Coords,
    ) -> FailResult<Coords>
    {Ok({
        trace!("============================");
        trace!("Begin relaxation # {}", iteration);

        let coords = do_cg_relax_with_param_optimization_if_supported(
            pot, &settings.cg, settings.parameters.as_ref(), coords, meta.sift(),
        )?;

        trace!("============================");

        let subdir = self.structure_path(EvLoopStructureKind::PreEvChase(iteration));
        self.write_stored_structure(
            &subdir,
            &format!("Structure after CG round {}", iteration),
            &coords, meta.sift(),
        )?;
        coords
    })}

    pub(in ::cmd) fn do_ev_loop_stuff_after_diagonalization(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        meta: HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
        iteration: Iteration,
        coords: Coords,
        evals: &Vec<f64>,
        evecs: &Basis3,
        allow_unfold_bands: bool,
    ) -> FailResult<(GammaSystemAnalysis, Coords, DidEvChasing)>
    {Ok({
        let classifications = super::acoustic_search::perform_acoustic_search(
            pot, evals, evecs,
            &coords, meta.sift(),
            &settings.acoustic_search,
        )?;
        trace!("Computing eigensystem info");
        let ev_analysis = super::run_gamma_system_analysis(
            &coords, meta.sift(), evals, evecs, Some(classifications),
            allow_unfold_bands,
        )?;
        {
            let file = self.create_file(format!("eigenvalues.{:02}", iteration))?;
            write_eigen_info_for_machines(&ev_analysis, file)?;
            write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;
        }
        let (coords, did_chasing) = self.maybe_do_ev_chasing(
            settings, pot, coords, meta.sift(), &ev_analysis, evals, evecs,
        )?;
        self.write_stored_structure(
            &self.structure_path(EvLoopStructureKind::PostEvChase(iteration)),
            &format!("Structure after eigenmode-chasing round {}", iteration),
            &coords, meta.sift(),
        )?;
        warn_on_improvable_lattice_params(pot, &coords, meta.sift())?;
        (ev_analysis, coords, did_chasing)
    })}

    fn maybe_do_ev_chasing(
        &self,
        settings: &Settings,
        pot: &PotentialBuilder,
        coords: Coords,
        meta: potential::CommonMeta,
        ev_analysis: &GammaSystemAnalysis,
        evals: &[f64],
        evecs: &Basis3,
    ) -> FailResult<(Coords, DidEvChasing)>
    {Ok({
        use super::acoustic_search::ModeKind;
        let bad_directions: Vec<_> = {
            let classifications = ev_analysis.ev_classifications.as_ref().expect("(bug) always computed!");
            izip!(1.., evals, &evecs.0, &classifications.0)
                .filter(|&(_, _, _, kind)| kind == &ModeKind::Imaginary)
                .map(|(i, freq, evec, _)| {
                    let name = format!("band {} ({})", i, freq);
                    let direction = EvDirection::from_eigenvector(evec, meta.sift());
                    (name, direction)
                }).collect()
        };

        match bad_directions.len() {
            0 => (coords, DidEvChasing(false)),
            n => {
                trace!("Chasing {} bad eigenvectors...", n);
                let structure = do_eigenvector_chase(
                    pot, &settings.ev_chase, coords, meta.sift(), &bad_directions[..],
                )?;
                (structure, DidEvChasing(true))
            }
        }
    })}
}

struct EvLoopFsm {
    config: cfg::EvLoop,
    iteration: Iteration,
    all_ok_count: u32,
}

pub enum EvLoopStatus {
    KeepGoing,
    Done,
    ItsBadGuys(&'static str),
}

// just a named bool for documentation
pub struct DidEvChasing(pub bool);

impl EvLoopFsm {
    pub fn new(config: &cfg::EvLoop) -> Self
    { EvLoopFsm {
        config: config.clone(),
        iteration: Iteration(1),
        all_ok_count: 0,
    }}

    pub fn step(&mut self, did: DidEvChasing) -> EvLoopStatus {
        self.iteration.0 += 1;
        match did {
            DidEvChasing(true) => {
                self.all_ok_count = 0;
                if self.iteration.0 > self.config.max_iter {
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

fn cg_builder_from_config(
    cg_settings: &cfg::Cg,
) -> (cg::Builder, cg::StopCondition) {
    let cfg::Cg {
        ref stop_condition, ref flavor, ref on_ls_failure,
        alpha_guess_first, alpha_guess_max,
    } = *cg_settings;

    let mut builder = match *flavor {
        cfg::CgFlavor::Acgsd { ls_iteration_limit } => {
            let mut builder = cg::Builder::new_acgsd();
            let mut ls_settings = rsp2_minimize::strong_ls::Settings::new();
            if let Some(value) = ls_iteration_limit {
                ls_settings.iteration_limit = value;
            }
            builder.linesearch(cg::settings::Linesearch::Acgsd(ls_settings));
            builder
        },
        cfg::CgFlavor::Hager {} => cg::Builder::new_hager(),
    };
    if let Some(x) = alpha_guess_first { builder.alpha_guess_first(x); }
    if let Some(x) = alpha_guess_max { builder.alpha_guess_max(x); }

    // FIXME XXX should not be a responsibility of the builder
    builder.on_ls_failure(match on_ls_failure {
        cfg::CgOnLsFailure::Succeed => cg::settings::OnLsFailure::Succeed,
        cfg::CgOnLsFailure::Fail => cg::settings::OnLsFailure::Fail,
        cfg::CgOnLsFailure::Warn => cg::settings::OnLsFailure::Warn,
    });

    (builder, stop_condition.clone())
}

fn do_cg_relax(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Cg,
    // NOTE: takes ownership of coords because it is likely an accident to reuse them
    coords: Coords,
    meta: potential::CommonMeta,
) -> FailResult<Coords>
{Ok({
    let mut flat_diff_fn = pot.parallel(true).initialize_flat_diff_fn(&coords, meta.sift())?;
    let relaxed_flat = {
        let (mut cg, stop_condition) = cg_builder_from_config(cg_settings);
        cg.stop_condition(stop_condition.to_function())
            .basic_output_fn(log_cg_output)
            .run(coords.to_carts().flat(), &mut *flat_diff_fn)
            .unwrap().position
    };
    coords.with_carts(relaxed_flat.nest().to_vec())
})}

fn log_cg_output(args: std::fmt::Arguments<'_>) { trace!("{}", args) }

//------------------

fn do_cg_relax_with_param_optimization_if_supported(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Cg,
    parameters: Option<&cfg::Parameters>,
    // NOTE: takes ownership of coords because it is likely an accident to reuse them
    coords: Coords,
    meta: potential::CommonMeta,
) -> FailResult<Coords>
{
    if let Some(parameters) = parameters {
        if let Some(x) = do_cg_relax_with_param_optimization(pot, cg_settings, parameters, &coords, meta.sift())? {
            return Ok(x);
        } else {
            trace!("Not relaxing with parameters because the potential does not support it.");
        }
    } else {
        trace!("Not relaxing with parameters because 'parameters' was not supplied.");
    }
    do_cg_relax(pot, cg_settings, coords, meta)
}

/// Returns Ok(None) if the potential does not support this method.
fn do_cg_relax_with_param_optimization(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Cg,
    parameters: &cfg::Parameters,
    coords: &Coords,
    meta: potential::CommonMeta,
) -> FailResult<Option<Coords>>
{Ok({
    let mut bond_diff_fn = match pot.parallel(true).initialize_bond_diff_fn(&coords, meta.sift())? {
        None => return Ok(None),
        Some(f) => f,
    };

    // Object that encapsulates the coordinate conversion logic for param optimization
    let helper = Rc::new(RelaxationOptimizationHelper::new(parameters, coords.lattice()));

    let (mut cg, stop_condition_cereal) = cg_builder_from_config(cg_settings);

    // Make the stop condition and output representative of the cartesian forces.
    cg.output_fn(get_param_opt_output_fn(helper.clone(), log_cg_output));
    cg.stop_condition({
        let helper = helper.clone();
        let mut imp = stop_condition_cereal.to_function();
        move |state: cg::AlgorithmState<'_>| {
            // HACK: to avoid code duplication, use the stop conditions built into rsp2_minimize,
            //       but feed them modified data.  I know that the stop condition won't look at
            //       most of these fields since I maintain both crates...          - ML
            let cg::AlgorithmState {
                iterations, value, position, gradient, direction, alpha,
                ..
            } = state;

            //
            let (d_carts, d_params) = helper.unflatten_grad(position, gradient);
            let mut gradient = d_carts.flat().to_vec();
            // FIXME HACK: include the parameter forces in the list so that they are checked
            //             by a "max-grad" constraint.
            // The proper solution is to make a new replacement for
            // rsp2_minimize::cg::stop_condition::Simple that has a "stress-max" variant.
            gradient.extend(d_params);

            let gradient = &gradient[..];
            imp(cg::AlgorithmState {
                iterations, value, position, gradient, direction, alpha,
                // HACK: To make matters even worse, we can't just replace the `gradient` field
                //       of state due to lifetime issues. We must construct a new one, and there
                //       is no public API for doing that (and I'm not sure I want to provide one).
                //
                //       So for now this field is `#[doc(hidden)] pub`...
                __no_full_destructure: (),
            })
        }
    });

    trace!("Incorporating parameter optimization into relaxation");
    let relaxed_flat = {
         cg.run(
            &helper.flatten_coords(&coords),
            {
                let helper = helper.clone();
                move |flat_coords| {
                    let ref coords = helper.unflatten_coords(flat_coords);
                    let (value, bond_grads) = bond_diff_fn.compute(coords, meta.clone())?;
                    let flat_grad = helper.flatten_grad(flat_coords, &bond_grads);
                    FailOk((value, flat_grad))
                }
            },
        ).unwrap().position
    };
    Some(helper.unflatten_coords(&relaxed_flat[..]))
})}

pub fn get_param_opt_output_fn(
    opt_helper: Rc<RelaxationOptimizationHelper>,
    mut emit: impl Clone + FnMut(std::fmt::Arguments<'_>) + 'static,
) -> impl Clone + FnMut(cg::AlgorithmState<'_>) + 'static {
    use rsp2_slice_math::vnorm;
    use std::fmt::Write;

    let mut dv_formatter = cg::output_fn_dv_formatter("dv:", 1);
    let mut cos_formatter = cg::output_fn_cosine_formatter("cos:", 3, 2);
    let opt_helper = opt_helper.clone();

    move |state: cg::AlgorithmState<'_>| {
        // Use cartesian gradient instead of actual gradient
        let (cart_grad, d_param) = opt_helper.unflatten_grad(state.position, state.gradient);
        let num_atoms = cart_grad.len();
        emit(format_args!(
            " i: {i:>6}  v: {v:7.3}  {dv}  g: {g:>6.1e}  max: {gm:>6.1e}  {fpar}  {cos}",
            i = state.iterations,
            v = state.value,
            dv = dv_formatter(state.value),
            g = vnorm(&cart_grad.flat()),
            gm = cart_grad.flat().iter().cloned().map(f64::abs).fold(0.0, f64::max),
            cos = cos_formatter(state.direction),
            // Force acting upon each lattice parameter, per atom.
            fpar = {
                let mut s = String::from("fpar:");
                for d_a in d_param {
                    let force_a = -d_a;
                    write!(s, " {:>+6.1e}", force_a / num_atoms as f64).unwrap();
                }
                s
            },
        ));
    }
}

//------------------

fn do_eigenvector_chase(
    pot: &PotentialBuilder,
    chase_settings: &cfg::EigenvectorChase,
    mut coords: Coords,
    meta: potential::CommonMeta,
    bad_directions: &[(String, EvDirection)],
) -> FailResult<Coords>
{Ok({
    match chase_settings {
        cfg::EigenvectorChase::OneByOne => {
            for (name, dir) in bad_directions {
                let dir = dir.as_real_checked();
                let (alpha, new_coords) = do_minimize_along_evec(pot, coords, meta.sift(), dir)?;
                info!("Optimized along {}, a = {:e}", name, alpha);

                coords = new_coords;
            }
            coords
        },
        cfg::EigenvectorChase::Cg(cg_settings) => {
            let bad_directions = bad_directions.iter().map(|(_, dir)| dir.clone());
            do_cg_along_evecs(pot, cg_settings, coords, meta.sift(), bad_directions)?
        },
    }
})}

fn do_cg_along_evecs(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Cg,
    coords: Coords,
    meta: potential::CommonMeta,
    directions: impl IntoIterator<Item=EvDirection>,
) -> FailResult<Coords>
{Ok({
    let directions = directions.into_iter().collect::<Vec<_>>();
    _do_cg_along_evecs(pot, cg_settings, coords, meta, &directions)?
})}

fn _do_cg_along_evecs(
    pot: &PotentialBuilder,
    cg_settings: &cfg::Cg,
    coords: Coords,
    meta: potential::CommonMeta,
    evecs: &[EvDirection],
) -> FailResult<Coords>
{Ok({
    let flat_evecs: Vec<_> = evecs.iter().map(|ev| ev.as_real_checked().flat()).collect();
    let init_pos = coords.to_carts();

    let mut flat_diff_fn = pot.parallel(true).initialize_flat_diff_fn(&coords, meta.sift())?;
    let relaxed_coeffs = {
        let (mut cg, stop_condition) = cg_builder_from_config(cg_settings);
        cg.stop_condition(stop_condition.to_function())
            .basic_output_fn(log_cg_output)
            .run(
                &vec![0.0; evecs.len()],
                &mut *constrained_diff_fn(&mut *flat_diff_fn, init_pos.flat(), &flat_evecs),
            ).unwrap().position
    };

    let final_flat_pos = flat_constrained_position(init_pos.flat(), &relaxed_coeffs, &flat_evecs);
    coords.with_carts(final_flat_pos.nest().to_vec())
})}

fn do_minimize_along_evec(
    pot: &PotentialBuilder,
    from_coords: Coords,
    meta: potential::CommonMeta,
    evec: &[V3],
) -> FailResult<(f64, Coords)>
{Ok({
    let mut diff_fn = pot.parallel(true).initialize_flat_diff_fn(&from_coords, meta.sift())?;

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
    meta: potential::CommonMeta,
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
