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
use crate::potential::{PotentialBuilder, DiffFn, BondDiffFn, DynCgDiffFn, CommonMeta};
use crate::meta::{self, prelude::*};
use crate::hlist_aliases::*;
use crate::math::basis::{GammaBasis3, EvDirection};
use crate::traits::{Save, AsPath};
use crate::util::ext_traits::PathNiceExt;

use super::trial::TrialDir;
use super::GammaSystemAnalysis;
use super::{write_eigen_info_for_humans, write_eigen_info_for_machines};
use super::{EvLoopStructureKind, Iteration};
use super::StopAfter;
use super::param_optimization::RelaxationOptimizationHelper;

use rsp2_tasks_config::{self as cfg, Settings};

use slice_of_array::prelude::*;
use rsp2_slice_math::{v, V, vdot};
use rsp2_array_types::{V3};
use rsp2_structure::{Coords};
use rsp2_minimize::{cg};
use rsp2_fs_util as fsx;

use std::rc::Rc;
use crate::filetypes::stored_structure;

impl TrialDir {
    /// NOTE: This writes to fixed filepaths in the trial directory
    ///       and is not designed to be called multiple times.
    pub(crate) fn do_main_ev_loop(
        &self,
        settings: &Settings,
        pot: &dyn PotentialBuilder,
        original_coords: Coords,
        meta: HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
        stop_after: StopAfter, // HACK
    ) -> FailResult<(Coords, Option<(GammaSystemAnalysis, Iteration)>)>
    {
        // `stop_after`, augmented with config sections required by those steps
        enum StopAfterPlus<'a> {
            Cg,
            Dynmat(&'a cfg::Phonons),
            DontStop(&'a cfg::Phonons),
        }

        let stop_after = match (stop_after, &settings.phonons) {
            (StopAfter::Cg, _) => StopAfterPlus::Cg,
            (StopAfter::Dynmat, Some(cfg)) => StopAfterPlus::Dynmat(cfg),
            (StopAfter::DontStop, Some(cfg)) => StopAfterPlus::DontStop(cfg),
            (StopAfter::Dynmat, None) |
            (StopAfter::DontStop, None) => bail!("`phonons:` config section is required"),
        };

        if !settings.ev_loop.enable {
            let iteration = None;
            let coords = self.do_ev_loop_stuff_before_dynmat(
                &settings, pot, meta.sift(), iteration, original_coords,
            )?;
            return Ok((coords, None));
        }

        let mut from_coords = original_coords;
        let mut loop_state = EvLoopFsm::new(&settings.ev_loop);
        loop {
            // move out of from_coords so that Rust's control-flow analysis
            // will make sure we put something back.
            let coords = from_coords;
            let iteration = loop_state.iteration;

            let coords = self.do_ev_loop_stuff_before_dynmat(
                &settings, pot, meta.sift(), Some(iteration), coords,
            )?;

            // rsp2-acgsd stops here
            let phonon_settings = match stop_after {
                StopAfterPlus::Cg => return Err(super::StoppedEarly.into()),
                StopAfterPlus::Dynmat(cfg) |
                StopAfterPlus::DontStop(cfg) => cfg,
            };

            let qpoint = V3::zero();
            let dynmat = super::do_compute_dynmat(
                Some(self), settings, phonon_settings, pot, qpoint, &coords, meta.sift(),
            )?;
            dynmat.save(self.gamma_dynmat_path(iteration))?;

            // rsp2-acgsd-and-dynmat stops here
            if let StopAfterPlus::Dynmat(_) = stop_after {
                return Err(super::StoppedEarly.into());
            }

            let (freqs, evecs) = pot.eco_mode(|eco_proof| {
                super::do_diagonalize_dynmat(phonon_settings, dynmat, eco_proof)
            })?;

            trace!("============================");
            trace!("Finished diagonalization");

            let (ev_analysis, coords, did_chasing) = {
                self.do_ev_loop_stuff_after_diagonalization(
                    &settings, pot, meta.sift(), iteration, coords, &freqs, &evecs,
                )?
            };

            match loop_state.step(did_chasing) {
                EvLoopStatus::KeepGoing => {
                    from_coords = coords;
                    continue;
                },
                EvLoopStatus::Done => {
                    return Ok((coords, Some((ev_analysis, iteration))));
                },
                EvLoopStatus::ItsBadGuys(msg) => {
                    bail!("{}", msg);
                },
            }
            // unreachable
        }
    }

    pub(in crate::cmd) fn do_ev_loop_stuff_before_dynmat(
        &self,
        settings: &Settings,
        pot: &dyn PotentialBuilder,
        meta: HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
        iteration: Option<Iteration>, // None when ev-loop is disabled
        coords: Coords,
    ) -> FailResult<Coords>
    {Ok({
        trace!("============================");
        match iteration {
            Some(iteration) => trace!("Begin relaxation # {}", iteration),
            None => trace!("Begin relaxation"),
        }

        let snapshot_fn = SnapshotFn::new(self.snapshot_structure_path(), meta.sift(), &settings.snapshot);
        let coords = do_cg_relax_with_param_optimization_if_supported(
            pot, &settings.cg, snapshot_fn, settings.parameters.as_ref(), coords, meta.sift(),
        )?;

        trace!("============================");

        if let Some(iteration) = iteration {
            let subdir = self.structure_path(EvLoopStructureKind::PreEvChase(iteration));
            self.write_stored_structure(
                &subdir,
                &format!("Structure after CG round {}", iteration),
                &coords, meta.sift(),
            )?;
        }
        coords
    })}

    pub(in crate::cmd) fn do_ev_loop_stuff_after_diagonalization(
        &self,
        settings: &Settings,
        pot: &dyn PotentialBuilder,
        meta: HList5<
            meta::SiteElements,
            meta::SiteMasses,
            Option<meta::SiteLayers>,
            Option<meta::LayerScMatrices>,
            Option<meta::FracBonds>,
        >,
        iteration: Iteration,
        coords: Coords,
        freqs: &Vec<f64>,
        evecs: &GammaBasis3,
    ) -> FailResult<(GammaSystemAnalysis, Coords, DidEvChasing)>
    {Ok({
        let classifications = super::acoustic_search::perform_acoustic_search(
            pot, freqs, evecs,
            &coords, meta.sift(),
            &settings.acoustic_search,
        )?;
        trace!("Computing eigensystem info");

        let unfold_bands = match settings.unfold_bands {
            None => false,
            Some(cfg::UnfoldBands::Zheng {}) => true,
        };

        let ev_analysis = super::do_gamma_system_analysis(
            &coords, meta.sift(), freqs, evecs, Some(classifications),
            unfold_bands,
        )?;
        {
            let file = self.create_file(format!("eigenvalues.{:02}", iteration))?;
            write_eigen_info_for_machines(&ev_analysis, file)?;
            write_eigen_info_for_humans(&ev_analysis, &mut |s| FailOk(info!("{}", s)))?;
        }

        let bad_directions: Vec<_> = {
            let classifications = ev_analysis.ev_classifications.as_ref().expect("(bug) always computed!");
            izip!(1.., freqs, &*evecs.0, &classifications.0)
                .filter(|&(_, _, _, kind)| kind == &super::acoustic_search::ModeKind::Imaginary)
                .map(|(i, &freq, evec, _)| {
                    let name = format!("band {} ({})", i, freq);
                    let direction = EvDirection::from_eigenvector(&evec.to_complex(), meta.sift());
                    (name, freq, direction)
                }).collect()
        };

        if let Some(animate_settings) = settings.animate.as_ref() {
            let all_guys = {
                zip_eq!(freqs, &*evecs.0)
                    .map(|(&freq, evec)| (freq, EvDirection::from_eigenvector(&evec.to_complex(), meta.sift())))
            };
            let bad_guys = {
                bad_directions.iter()
                    .map(|&(_, freq, ref dir)| (freq, dir.clone()))
            };
            let result = self.write_animations(
                animate_settings,
                &coords, meta.sift(), iteration,
                bad_guys, all_guys,
            );

            // not the end of the world if this failed
            if let Err(e) = result {
                warn!("Error while writing animations: {}", e);
            }
        }

        let (coords, did_chasing) = {
            match bad_directions.len() {
                0 => (coords, DidEvChasing(false)),
                n => {
                    trace!("Chasing {} bad eigenvectors...", n);
                    let structure = do_eigenvector_chase(
                        pot, &settings.ev_chase, coords, meta.sift(), &bad_directions[..],
                    )?;
                    (structure, DidEvChasing(true))
                },
            }
        };
        self.write_stored_structure(
            &self.structure_path(EvLoopStructureKind::PostEvChase(iteration)),
            &format!("Structure after eigenmode-chasing round {}", iteration),
            &coords, meta.sift(),
        )?;
        warn_on_improvable_lattice_params(pot, &coords, meta.sift())?;
        (ev_analysis, coords, did_chasing)
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
    builder.alpha_guess_first(alpha_guess_first);
    builder.alpha_guess_max(alpha_guess_max);

    // FIXME XXX should not be a responsibility of the builder
    builder.on_ls_failure(match on_ls_failure {
        cfg::CgOnLsFailure::Succeed => cg::settings::OnLsFailure::Succeed,
        cfg::CgOnLsFailure::Fail => cg::settings::OnLsFailure::Fail,
        cfg::CgOnLsFailure::Warn => cg::settings::OnLsFailure::Warn,
    });

    (builder, stop_condition.clone())
}

fn do_cg_relax(
    pot: &dyn PotentialBuilder,
    cg_settings: &cfg::Cg,
    snapshot_fn: SnapshotFn,
    // NOTE: takes ownership of coords because it is likely an accident to reuse them
    coords: Coords,
    meta: CommonMeta,
) -> FailResult<Coords>
{Ok({
    let mut flat_diff_fn = pot.parallel(true).initialize_cg_diff_fn(&coords, meta.sift())?;
    let unflatten_coords = {
        let coords = coords.clone();
        move |flat: &[f64]| coords.with_carts(flat.nest().to_vec())
    };

    let relaxed_flat = {
        let (mut cg, stop_condition) = cg_builder_from_config(cg_settings);
        cg.stop_condition(stop_condition.to_function())
            .basic_output_fn(log_cg_output)
            .output_fn({
                let unflatten_coords = unflatten_coords.clone();
                move |state: cg::AlgorithmState<'_>| {
                    snapshot_fn.maybe_save_snapshot(&state, unflatten_coords(state.position))
                }
            })
            .run(coords.to_carts().flat(), &mut *flat_diff_fn)
            .unwrap().position
    };
    unflatten_coords(&relaxed_flat)
})}

fn log_cg_output(args: std::fmt::Arguments<'_>) { trace!("{}", args) }

//------------------

fn do_cg_relax_with_param_optimization_if_supported(
    pot: &dyn PotentialBuilder,
    cg_settings: &cfg::Cg,
    snapshot_fn: SnapshotFn,
    parameters: Option<&cfg::Parameters>,
    // NOTE: takes ownership of coords because it is likely an accident to reuse them
    coords: Coords,
    meta: CommonMeta,
) -> FailResult<Coords>
{
    if let Some(parameters) = parameters {
        if let Some(x) = do_cg_relax_with_param_optimization(pot, cg_settings, snapshot_fn.clone(), parameters, &coords, meta.sift())? {
            return Ok(x);
        } else {
            trace!("Not relaxing with parameters because the potential does not support it.");
        }
    } else {
        trace!("Not relaxing with parameters because 'parameters' was not supplied.");
    }
    do_cg_relax(pot, cg_settings, snapshot_fn, coords, meta)
}

/// Returns Ok(None) if the potential does not support this method.
fn do_cg_relax_with_param_optimization(
    pot: &dyn PotentialBuilder,
    cg_settings: &cfg::Cg,
    snapshot_fn: SnapshotFn,
    parameters: &cfg::Parameters,
    coords: &Coords,
    meta: CommonMeta,
) -> FailResult<Option<Coords>>
{Ok({
    let bond_diff_fn = match pot.parallel(true).initialize_bond_diff_fn(&coords, meta.sift())? {
        None => return Ok(None),
        Some(f) => f,
    };

    // Object that encapsulates the coordinate conversion logic for param optimization
    let param_helper = Rc::new(RelaxationOptimizationHelper::new(parameters, coords.lattice()));

    let (mut cg, stop_condition_cereal) = cg_builder_from_config(cg_settings);

    // Make the stop condition and output representative of the cartesian forces.
    cg.output_fn(get_param_opt_output_fn(param_helper.clone(), log_cg_output));
    cg.output_fn({
        let param_helper = param_helper.clone();
        move |state: cg::AlgorithmState<'_>| {
            snapshot_fn.maybe_save_snapshot(&state, param_helper.unflatten_coords(state.position))
        }
    });
    cg.stop_condition({
        let param_helper = param_helper.clone();
        let mut stop_condition_imp = stop_condition_cereal.to_function();
        move |state: cg::AlgorithmState<'_>| {
            // HACK: to avoid code duplication, use the stop conditions built into rsp2_minimize,
            //       but feed them modified data.  I know that the stop condition won't look at
            //       most of these fields since I maintain both crates...          - ML
            let cg::AlgorithmState {
                iterations, value, position, gradient, direction, alpha,
                ..
            } = state;

            let (d_carts, d_params) = param_helper.unflatten_grad(position, gradient);
            let mut gradient = d_carts.flat().to_vec();
            // FIXME HACK: include the parameter forces in the list, scaled to an intrinsic
            //             quantity, so that they are checked by a "max-grad" constraint.
            // The proper solution is to make a new replacement for
            // rsp2_minimize::cg::stop_condition::Simple that has a "stress-max" variant.
            gradient.extend(d_params.into_iter().map(|x| x / d_carts.len() as f64));

            let gradient = &gradient[..];
            stop_condition_imp(cg::AlgorithmState {
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
            &param_helper.flatten_coords(&coords),
            {
                struct Adapter {
                    helper: Rc<RelaxationOptimizationHelper>,
                    bond_diff_fn: Box<dyn BondDiffFn<CommonMeta>>,
                    meta: CommonMeta,
                }

                impl cg::DiffFn for Adapter {
                    type Error = failure::Error;

                    fn compute(&mut self, flat_coords: &[f64]) -> Result<(f64, Vec<f64>), failure::Error> {
                        let Adapter { ref helper, ref mut bond_diff_fn, ref meta } = *self;
                        let ref coords = helper.unflatten_coords(flat_coords);
                        let (value, bond_grads) = bond_diff_fn.compute(coords, meta.clone())?;
                        let flat_grad = helper.flatten_grad(flat_coords, &bond_grads);
                        Ok((value, flat_grad))
                    }

                    fn check(&mut self, flat_coords: &[f64]) -> Result<(), failure::Error> {
                        let Adapter { ref helper, ref mut bond_diff_fn, ref meta } = *self;
                        let ref coords = helper.unflatten_coords(flat_coords);
                        bond_diff_fn.check(coords, meta.clone())
                    }
                }

                let helper = param_helper.clone();
                Adapter { helper, bond_diff_fn, meta }
            },
        ).unwrap().position
    };
    Some(param_helper.unflatten_coords(&relaxed_flat[..]))
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
            " i: {i:>6}  v: {v:7.3}  {dv}  g: {g:>6.1e}  max: {gm:>6.1e}  {fpar}  α: {a:>6.1e} {cos}",
            i = state.iterations,
            v = state.value,
            dv = dv_formatter(state.value),
            g = vnorm(&cart_grad.flat()),
            a = state.alpha,
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

// because `impl Clone + FnMut(&cg::AlgorithmState<'_>, Box<dyn FnOnce() -> &Coords>) + 'static`
// is a mouthful.
#[derive(Debug, Clone)]
struct SnapshotFn {
    path: std::path::PathBuf,
    settings: cfg::Snapshot,
    meta: stored_structure::Meta,
}

impl SnapshotFn {
    fn new(
        path: impl AsPath,
        meta: stored_structure::Meta,
        settings: &cfg::Snapshot,
    ) -> Self {
        let path = path.as_path().to_owned();
        let settings = settings.clone();
        Self { path, settings, meta }
    }

    fn maybe_save_snapshot(
        &self,
        state: &cg::AlgorithmState<'_>,
        coords: Coords,
    ) {
        let period = match self.settings.every {
            None | Some(0) => return,
            Some(period) => period,
        };

        if state.iterations > 0 && (state.iterations - 1) % period as u64 == 0 {
            if let Err(e) = fsx::rm_rf(&self.path) {
                warn_once!("failed to delete old snapshot at {}: {}", self.path.nice(), e);
            }

            let title = format!("Snapshot after {} CG iterations", state.iterations);
            let structure = stored_structure::StoredStructure::from_parts(
                title, coords, self.meta.clone(),
            );
            if let Err(e) = structure.save(&self.path) {
                warn_once!("failed to write snapshot at {}: {}", self.path.nice(), e);
            }
        }
    }
}


//------------------

fn do_eigenvector_chase(
    pot: &dyn PotentialBuilder,
    chase_settings: &cfg::EigenvectorChase,
    mut coords: Coords,
    meta: CommonMeta,
    bad_directions: &[(String, f64, EvDirection)],
) -> FailResult<Coords>
{Ok({
    match chase_settings {
        cfg::EigenvectorChase::OneByOne => {
            for (name, _, dir) in bad_directions {
                let dir = dir.as_real_checked();
                let (alpha, new_coords) = do_minimize_along_evec(pot, coords, meta.sift(), dir)?;
                info!("Optimized along {}, a = {:e}", name, alpha);

                coords = new_coords;
            }
            coords
        },
        cfg::EigenvectorChase::Cg(cg_settings) => {
            let bad_directions = bad_directions.iter().map(|(_, _, dir)| dir.clone());
            do_cg_along_evecs(pot, cg_settings, coords, meta.sift(), bad_directions)?
        },
    }
})}

fn do_cg_along_evecs(
    pot: &dyn PotentialBuilder,
    cg_settings: &cfg::Cg,
    coords: Coords,
    meta: CommonMeta,
    directions: impl IntoIterator<Item=EvDirection>,
) -> FailResult<Coords>
{Ok({
    let directions = directions.into_iter().collect::<Vec<_>>();
    _do_cg_along_evecs(pot, cg_settings, coords, meta, &directions)?
})}

fn _do_cg_along_evecs(
    pot: &dyn PotentialBuilder,
    cg_settings: &cfg::Cg,
    coords: Coords,
    meta: CommonMeta,
    evecs: &[EvDirection],
) -> FailResult<Coords>
{Ok({
    let flat_evecs: Vec<_> = evecs.iter().map(|ev| ev.as_real_checked().flat()).collect();
    let init_pos = coords.to_carts();

    let mut flat_diff_fn = pot.parallel(true).initialize_cg_diff_fn(&coords, meta.sift())?;
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
    pot: &dyn PotentialBuilder,
    from_coords: Coords,
    meta: CommonMeta,
    evec: &[V3],
) -> FailResult<(f64, Coords)>
{Ok({
    let mut diff_fn = pot.parallel(true).initialize_cg_diff_fn(&from_coords, meta.sift())?;

    let direction = &evec[..];
    let from_pos = from_coords.to_carts();
    let pos_at_alpha = |alpha| {
        let V(pos) = v(from_pos.flat()) + alpha * v(direction.flat());
        pos
    };
    let alpha = rsp2_minimize::exact_ls(0.0, 1e-4, |alpha| {
        let gradient = diff_fn.compute(&pos_at_alpha(alpha))?.1;
        let slope = vdot(&gradient[..], direction.flat());
        FailOk(::rsp2_minimize::exact_ls::Slope(slope))
    })??.alpha;
    let pos = pos_at_alpha(alpha);

    (alpha, from_coords.with_carts(pos.nest().to_vec()))
})}

fn warn_on_improvable_lattice_params(
    pot: &dyn PotentialBuilder,
    coords: &Coords,
    meta: CommonMeta,
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
    flat_3n_diff_fn: &'a mut DynCgDiffFn<'a>,
    // K values, K <= 3N
    flat_init_pos: &'a [f64],
    // K eigenvectors
    flat_evs: &'a [&[f64]],
) -> Box<DynCgDiffFn<'a>>
{
    struct Adapter<'b> {
        flat_init_pos: &'b [f64],
        flat_3n_diff_fn: &'b mut DynCgDiffFn<'b>,
        flat_evs: &'b [&'b [f64]],
    }

    impl<'b> cg::DiffFn for Adapter<'b> {
        type Error = failure::Error;

        fn compute(&mut self, coeffs: &[f64]) -> FailResult<(f64, Vec<f64>)>
        {Ok({
            let Adapter { flat_init_pos, flat_3n_diff_fn, flat_evs } = self;

            // This is dead simple.
            // The kth element of the new gradient is the slope along the kth ev.
            // The change in position is a sum over contributions from each ev.
            // These relationships have a simple expression in terms of
            //   the matrix whose columns are the selected eigenvectors.
            // (though the following is transposed for our row-centric formalism)
            let flat_pos = flat_constrained_position(flat_init_pos, coeffs, flat_evs);

            let (value, flat_grad) = flat_3n_diff_fn.compute(&flat_pos)?;

            let grad = dot_mat_vec_dumb(flat_evs, &flat_grad);
            (value, grad)
        })}

        fn check(&mut self, coeffs: &[f64]) -> FailResult<()> {
            let Adapter { flat_init_pos, flat_3n_diff_fn, flat_evs } = self;

            let flat_pos = flat_constrained_position(flat_init_pos, coeffs, flat_evs);

            flat_3n_diff_fn.check(&flat_pos)
        }
    }
    Box::new(Adapter { flat_init_pos, flat_3n_diff_fn, flat_evs })
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
