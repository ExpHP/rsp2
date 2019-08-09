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

use crate::FailResult;
use crate::VersionInfo;
use crate::cmd::trial::{TrialDir, NewTrialDirArgs};
use crate::cmd::{StructureFileType, DidEvChasing};
use crate::traits::{Save, Load};
use crate::ui::logging::{init_global_logger, SetGlobalLogfile};
use crate::ui::cfg_merging::ConfigSources;
use crate::ui::cli_deserialize::CliDeserialize;
use crate::util::ext_traits::{ArgMatchesExt};
use crate::filetypes::{StoredStructure, Eigensols};

use rsp2_tasks_config as cfg;
use rsp2_lammps_wrap::LammpsOnDemand;

use cfg::{ValidatedSettings, ValidatedEnergyPlotSettings};

use clap;
use path_abs::{PathDir, PathFile, PathAbs};
use std::ffi::OsStr;
use std::panic::UnwindSafe;
use std::process::exit;

// -------------------------------------------------------------------------------------
// Initialization common to all entry points

fn wrap_main<F>(version: VersionInfo, main: F) -> !
where
    F: FnOnce(SetGlobalLogfile, Option<LammpsOnDemand>) -> FailResult<()>,
    F: UnwindSafe,
{
    wrap_main_with_lammps_on_demand(|on_demand| {
        // From here onwards, everything runs on only a single process.
        let result = (|| { // scope '?'
            let logfile = init_global_logger().expect("Could not init logger");
            log_version(version);
            check_for_deps()?;
            log_thread_info()?;

            let _pp_guard = rsp2_python::add_to_python_path()?;

            main(logfile, on_demand)
        })();

        result.unwrap_or_else(|e| {
            // HACK
            if let Some(crate::cmd::StoppedAfterDynmat) = e.downcast_ref() {
                return;
            }

            show_errors(e);
            exit(1);
        });
    });
}

fn wrap_main_just_for_ui<F>(main: F) -> !
where
    F: FnOnce(SetGlobalLogfile) -> FailResult<()>,
    F: UnwindSafe,
{
    // FIXME: copy-pasta due to ordering of init code in `wrap_main` making these bits
    //        difficult to factor out.
    let result = {
        let logfile = init_global_logger().expect("Could not init logger");
        main(logfile)
    };

    result.unwrap_or_else(|e| {
        show_errors(e);
        exit(1);
    });
    exit(0);
}

// This initializes MPI so it must be done at the very beginning.
//
// The closure runs on only one process.
fn wrap_main_with_lammps_on_demand(continuation: impl UnwindSafe + FnOnce(Option<LammpsOnDemand>)) -> ! {
    #[cfg(feature = "mpi")] {
        let required = mpi::Threading::Serialized;
        let (_universe, actual) = {
            mpi::initialize_with_threading(required).expect("Could not initialize MPI!")
        };

        // 'actual >= required' would be nicer, but I don't think MPI specifies comparison ordering
        assert_eq!(actual, required);

        LammpsOnDemand::with_mpi_abort_on_unwind(|| {
            LammpsOnDemand::install(|on_demand| continuation(Some(on_demand)));
        });

        // NOTE: drop of _universe here issues MPI_Finalize
    }
    #[cfg(not(feature = "mpi"))] {
        continuation(None);
    }
    exit(0)
}

// The commit is useful to have in logfiles since rsp2 has no concept of "releases".
fn log_version(version: VersionInfo) {
    info!("rsp2 ({} {})", version.short_sha, version.commit_date);
}

fn log_thread_info() -> FailResult<()> {
    info!("Available resources for parallelism:");

    #[cfg(feature = "mpi")] {
        info!("    MPI: {} process(es)", crate::env::num_mpi_processes());
    }
    #[cfg(not(feature = "mpi"))] {
        info!("    MPI: N/A (disabled during compilation)");
    }

    // Currently, rsp2 exposes the same value of OMP_NUM_THREADS to both Lammps (which creates
    // them per process) and to python (which is only run on one process),
    // so OMP info is deliberately vague and currently only here for debugging.
    info!(
        " OpenMP: {} thread(s) per process ({})",
        crate::env::omp_num_threads()?,
        crate::env::OMP_NUM_THREADS,
    );
    info!(
        "       : {} thread(s) in single-process tasks ({})",
        crate::env::max_omp_num_threads()?,
        crate::env::MAX_OMP_NUM_THREADS,
    );
    info!("  rayon: {} thread(s) on the root process", rayon::current_num_threads());
    Ok(())
}

fn show_errors(e: failure::Error) {
    for cause in e.iter_chain() {
        error!("{}", cause);
    }

    if std::env::var_os("RUST_BACKTRACE") == Some(OsStr::new("1").to_owned()) {
        error!("{}", e.backtrace());
    } else {
        // When the only user is also the only dev, there isn't much point to wrapping
        // error messages in context.  As a result of this, some error messages are
        // *particularly* terrible.  (e.g. "cannot parse integer from empty string"
        // without any indication of which file caused it).
        //
        // For now, leave a reminder about RUST_BACKTRACE.
        error!("\
            (If you found the above error message to be particularly lacking in \
            detail, try again with RUST_BACKTRACE=1)\
        ");
    }
}

fn check_for_deps() -> FailResult<()> {
    crate::cmd::python::check_availability()?;

    if std::env::var_os("LAMMPS_POTENTIALS").is_none() {
        bail!("rsp2 requires you to set the LAMMPS_POTENTIALS environment variable.");
    }

    Ok(())
}

// -------------------------------------------------------------------------------------
// Some commonly used CLI args.

// Configuration YAML obtained from CLI args, for an initial run.
struct ConfigArgs(ConfigSources);

// Configuration YAML obtained from CLI args, for future runs.
// (disables the requirement for having at least one)
struct ConfigOverrideArgs(Option<ConfigSources>);

impl CliDeserialize for ConfigArgs {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!(*config [-c][--config]=CONFIG... crate::ui::cfg_merging::CONFIG_HELP_STR),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches<'_>) -> FailResult<Self>
    { ConfigSources::resolve_from_args(m.expect_values_of("config")).map(ConfigArgs) }
}

impl CliDeserialize for ConfigOverrideArgs {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!(?config [-c][--config]=CONFIG... crate::ui::cfg_merging::CONFIG_OVERRIDE_HELP_STR),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches<'_>) -> FailResult<Self> {
        if let Some(args) = m.values_of("config") {
            Ok(ConfigOverrideArgs(Some(ConfigSources::resolve_from_args(args)?)))
        } else {
            Ok(ConfigOverrideArgs(None))
        }
    }
}

impl CliDeserialize for NewTrialDirArgs {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        let app = app.args(&[
            arg!(*trial_dir [-o][--output]=OUTDIR "output trial directory"),
            arg!( force [-f][--force] "replace existing output directories"),
        ]);
        ConfigArgs::_augment_clap_app(app)
    }

    fn _resolve_args(m: &clap::ArgMatches<'_>) -> FailResult<Self>
    { Ok(NewTrialDirArgs {
        config_sources: ConfigArgs::_resolve_args(m)?.0,
        err_if_existing: !m.is_present("force"),
        // FIXME factor out 'absolute()'
        trial_dir: PathDir::current_dir()?.as_path().join(m.expect_value_of("trial_dir")),
    })}
}

// (not sure why `impl CliDeserialize for Option<StructureFileType>` isn't good enough
//  but rustc says Option<_> doesn't impl CliDeserialize, even when it ought to be
//  inferrable that the _ is StructureFileType)
pub struct OptionalFileType(Option<StructureFileType>);

impl CliDeserialize for OptionalFileType {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!( structure_type [--structure-type]=STYPE "Structure filetype. \
                [choices: poscar, layers, guess] [default: guess]\
            "),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches<'_>) -> FailResult<Self> {
        Ok(OptionalFileType({
            if let Some(s) = m.value_of("structure_type") {
                match s {
                    "poscar" => Some(StructureFileType::Poscar),
                    "layers" => Some(StructureFileType::LayersYaml),
                    "dir"    => Some(StructureFileType::StoredStructure),
                    "guess" => None,
                    _ => bail!("invalid setting for --structure-type"),
                }
            } else { None }
        }))
    }
}

impl OptionalFileType {
    pub fn or_guess(self, path: &PathAbs) -> StructureFileType {
        let default = StructureFileType::Poscar;

        self.0.unwrap_or_else(|| {
            let meta = match path.as_path().metadata() {
                Ok(meta) => meta,
                Err(_) => return default,
            };

            if meta.is_file() {
                match path.as_path().extension().and_then(|s| s.to_str()) {
                    Some("yaml") => StructureFileType::LayersYaml,
                    Some("vasp") => StructureFileType::Poscar,
                    Some("xyz") => StructureFileType::Xyz,
                    _ => default,
                }
            } else if meta.is_dir() {
                StructureFileType::StoredStructure
            } else {
                default
            }
        })
    }
}

/// Used by entry points that have no trial directory, but where the user may wish to save
/// log output to a specified path, at their choosing.
pub struct AppendLog(pub AppendLogInner);

pub struct AppendLogInner {
    path: Option<String>,
}

impl CliDeserialize for AppendLog {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!( log [--log]=LOGFILE "append to this logfile"),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches<'_>) -> FailResult<Self> {
        Ok(AppendLog(AppendLogInner {
            path: m.value_of("log").map(|s| s.to_string()),
        }))
    }
}

impl AppendLogInner {
    fn start(self, logfile: SetGlobalLogfile) -> FailResult<()> {
        if let Some(path) = self.path {
            logfile.start(PathFile::create(path)?)?;
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------------------

// %% CRATES: binary: rsp2 %%
pub fn rsp2(bin_name: &str, version: VersionInfo) -> ! {
    _rsp2_acgsd(false, bin_name, version)
}

// HACK
// %% CRATES: binary: rsp2-acgsd-and-dynmat %%
pub fn rsp2_acgsd_and_dynmat(bin_name: &str, version: VersionInfo) -> ! {
    _rsp2_acgsd(true, bin_name, version)
}

fn _rsp2_acgsd(
    stop_after_dynmat: bool,
    bin_name: &str,
    version: VersionInfo,
) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .about("runs the full eigenvector loop of rsp2")
                .args(&[
                    arg!( input=STRUCTURE "input file for structure"),
                ])
        });
        let matches = app.get_matches();
        let (dir_args, filetype) = de.resolve_args(&matches)?;

        let input = PathAbs::new(matches.expect_value_of("input"))?;
        let filetype = OptionalFileType::or_guess(filetype, &input);

        let mut trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let ValidatedSettings(settings) = trial.read_base_settings()?;
        trial.run_relax_with_eigenvectors(mpi_on_demand, &settings, filetype, &input, stop_after_dynmat)
    });
}

// %% CRATES: binary: rsp2-after-diagonalization %%
pub fn after_diagonalization(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .about("The next step after rsp2-acgsd-and-dynmat and the negative_modes py script")
                .args(&[
                    arg!( diagonalize [--diagonalize] "\
                        perform diagonalization, rather than expecting ev-loop-modes-NN.json.gz to \
                        exist. This is provided for cases involving the dense diagonalizer on \
                        large systems where the python script may require too much memory.\
                    "),
                    arg!( dir=DIR "existing trial directory"),
                ])
        });
        let matches = app.get_matches();
        let ConfigOverrideArgs(overrides) = de.resolve_args(&matches)?;

        let dir = PathDir::new(matches.expect_value_of("dir"))?;
        let mut trial = TrialDir::from_existing(dir.as_path())?;

        // Make sure the run is valid before making a logfile
        let will_diagonalize = matches.is_present("diagonalize");
        let iteration = trial.find_iteration_for_ev_chase(will_diagonalize)?;

        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let ValidatedSettings(settings) = match overrides {
            Some(overrides) => {
                let save_path = trial.modified_settings_path(iteration);
                trial.read_modified_settings(overrides, Some(&save_path))?
            },
            None => trial.read_base_settings()?,
        };

        let DidEvChasing(chased) = trial.run_after_diagonalization(
            mpi_on_demand, &settings, iteration, will_diagonalize,
        )?;
        match chased {
            true => bail!("Ev chasing was performed; loop not done"),
            false => Ok(())
        }
    });
}

// %% CRATES: binary: rsp2-shear-plot %%
pub fn shear_plot(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .args(&[
                    arg!( input=STRUCTURE "existing structure directory"),
                    arg!(*output [--output][-o]=OUTPUT "output JSON file"),
                ])
        });
        let matches = app.get_matches();
        let ((ConfigArgs(config), AppendLog(append_log)), plot_args) = de.resolve_args(&matches)?;

        append_log.start(logfile)?;

        let input = StoredStructure::load(matches.expect_value_of("input"))?;
        let output = matches.expect_value_of("output");

        let ValidatedEnergyPlotSettings(settings) = config.deserialize()?;
        crate::cmd::run_shear_plot(mpi_on_demand, &settings, input, plot_args, output)
    });
}

// %% CRATES: binary: rsp2-rerun-analysis %%
pub fn rerun_analysis(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .args(&[
                    arg!( dir=DIR "existing trial directory, or a structure directory within one"),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        let dir = PathDir::new(matches.expect_value_of("dir"))?;
        let (mut trial, structure) = crate::cmd::resolve_trial_or_structure_path(dir.as_ref(), "final.structure")?;

        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let ValidatedSettings(settings) = trial.read_base_settings()?;
        trial.rerun_ev_analysis(mpi_on_demand, &settings, structure)
    });
}

// FIXME it is kind of dumb having both this and rerun-analysis.
//       the trouble is they take different input; rerun-analysis rediagonalizes the entire
//       system using phonopy (and thus needs access to settings to know the potential),
//       while this requires the eigensolutions as input.
// %% CRATES: binary: rsp2-sparse-analysis %%
pub fn sparse_analysis(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, _mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .args(&[
                    arg!( structure=STRUCTUREDIR "structure in directory format"),
                    arg!( eigensols=EIGENSOLS "eigensolutions file"),
                    arg!(*output [--output][-o]=EIGENSOLS "\
                        output directory. This can be the trial directory. Existing analysis \
                        output files will be clobbered.\
                    "),
                ])
        });
        let matches = app.get_matches();
        let AppendLog(append_log) = de.resolve_args(&matches)?;
        append_log.start(logfile)?;

        let structure = StoredStructure::load(matches.expect_value_of("structure"))?;

        let (freqs, evecs) = {
            let path = PathFile::new(matches.expect_value_of("eigensols"))?;
            let Eigensols { frequencies, eigenvectors } = Load::load(path)?;
            let eigenvectors = eigenvectors.into_gamma_basis3().ok_or_else(|| {
                failure::err_msg("expected real eigensols!")
            })?;
            (frequencies, eigenvectors)
        };

        // reminder: does not fail on existing
        let outdir = PathDir::create(matches.expect_value_of("output"))?;

        let analysis = crate::cmd::run_sparse_analysis(structure, &freqs, &evecs)?;

        crate::cmd::write_ev_analysis_output_files(&outdir, &analysis)?;
        Ok(())
    });
}

// FIXME yet another *-analysis entry point.
//       This one takes the dynamical matrix as input, and requires access to the settings file.
// %% CRATES: binary: rsp2-dynmat-analysis %%
pub fn dynmat_analysis(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .args(&[
                    arg!( structure=STRUCTUREDIR "structure in directory format"),
                    arg!( dynmat=DYNMAT "dynmat.npz"),
                    arg!(*output [--output][-o]=OUTDIR "\
                        output directory. This can be the trial directory. Existing analysis \
                        output files will be clobbered.\
                    "),
                ])
        });
        let matches = app.get_matches();
        let (ConfigArgs(config), AppendLog(append_log)) = de.resolve_args(&matches)?;
        append_log.start(logfile)?;

        let structure = StoredStructure::load(matches.expect_value_of("structure"))?;

        let dynmat = Load::load(PathFile::new(matches.expect_value_of("dynmat"))?)?;

        // reminder: does not fail on existing
        let outdir = PathDir::create(matches.expect_value_of("output"))?;

        let ValidatedSettings(settings) = config.deserialize()?;

        let analysis = crate::cmd::run_dynmat_analysis(&settings, structure, mpi_on_demand, dynmat)?;

        crate::cmd::write_ev_analysis_output_files(&outdir, &analysis)?;
        Ok(())
    });
}

// %% CRATES: binary: rsp2-bond-test %%
pub fn bond_test(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, _mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .args(&[
                    arg!( input=STRUCTURE ""),
                    arg!( cart [--cart] "output CartBonds instead of FracBonds")
                ])
        });
        let matches = app.get_matches();
        let filetype = de.resolve_args(&matches)?;

        logfile.disable();

        let input = PathAbs::new(matches.expect_value_of("input"))?;
        let filetype = OptionalFileType::or_guess(filetype, &input);

        let (coords, _) = crate::cmd::read_optimizable_structure(None, None, filetype, &input)?;
        let coords = coords.construct(); // reckless

        let bonds = rsp2_structure::bonds::FracBonds::compute(&coords, 1.8)?;
        match matches.is_present("cart") {
            true => serde_json::to_writer(::std::io::stdout(), &bonds.to_cart_bonds(&coords))?,
            false => serde_json::to_writer(::std::io::stdout(), &bonds)?,
        }

        println!(); // flush, dammit
        Ok(())
    });
}

// %% CRATES: binary: rsp2-plot-vdw %%
pub fn plot_vdw(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .args(&[
                    arg!(*z [-z]=ZSEP "z separation"),
                    arg!( r_min [--r-min]=RMIN ""),
                    arg!( r_max [--r-max]=RMAX ""),
                    arg!( steps [--steps]=RSTEP ""),
                ])
        });
        let matches = app.get_matches();
        let ConfigArgs(config) = de.resolve_args(&matches)?;

        logfile.disable(); // no trial dir

        let z: f64 = matches.expect_value_of("z").parse()?;
        let r_min: f64 = matches.value_of("r_min").map_or(Ok(z), str::parse)?;
        let r_max: f64 = matches.value_of("r_max").unwrap_or("15.0").parse()?;
        let steps: u32 = matches.value_of("steps").unwrap_or("200").parse()?;

        let rs = (0..steps).map(|i| {
            let alpha = i as f64 / (steps as f64 - 1.0);
            r_min * (1.0 - alpha) + r_max * alpha
        }).collect::<Vec<_>>();

        crate::cmd::run_plot_vdw(mpi_on_demand, &config.deserialize()?, z, &rs[..])
    });
}

// %% CRATES: binary: rsp2-converge-vdw %%
pub fn converge_vdw(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .args(&[
                    arg!(*z [-z]=ZSEP "z separation"),
                    arg!( r_min [--r-min]=RMIN ""),
                    arg!( r_max [--r-max]=RMAX ""),
                    arg!( steps [--steps]=RSTEP ""),
                ])
        });
        let matches = app.get_matches();
        let ConfigArgs(config) = de.resolve_args(&matches)?;

        logfile.disable(); // no trial dir

        let z: f64 = matches.expect_value_of("z").parse()?;
        let r_min: f64 = matches.value_of("r_min").map_or(Ok(z), str::parse)?;
        let r_max: f64 = matches.value_of("r_max").unwrap_or("15.0").parse()?;

        crate::cmd::run_converge_vdw(mpi_on_demand, &config.deserialize()?, z, (r_min, r_max))
    });
}

// %% CRATES: binary: rsp2-make-supercell %%
pub fn make_supercell(bin_name: &str, _version: VersionInfo) -> ! {
    wrap_main_just_for_ui(|logfile| {
        use crate::cmd::LayerScMode;

        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .about("Makes a supercell of a structure that is in rsp2's 'directory.structure' format.")
                .args(&[
                    arg!( dims=SC_MATRIX "\
                        A JSON literal of an integer, integer vector, or matrix. \
                        Each row of the matrix describes a basis vector of the output as a linear \
                        combination of the basis vectors in the original lattice. \
                        A vector is interpreted as a diagonal matrix, and an integer is interpreted \
                        as multiplied by the identity.\
                    "),
                    arg!( primitive [--primitive]=MODE "\
                        Choices: [keep, input, auto, none]. Default: auto. Determines how the \
                        'layer_sc_matrices' field of the output is written, which is used for \
                        unfolding. 'keep' requires the input to contain layer SC matrices, and \
                        will preserve the original primitive cells by multiplying against those \
                        matrices. 'input' defines the input structure to be the the primitive by \
                        setting all SC matrices equal to 'dims'. 'auto' is equivalent to either \
                        'keep' or 'input' based on whether the input contains SC matrices.\
                    "),
                    arg!( input=INPUT_DIR ""),
                    arg!(*output [-o][--output]=PATH ""),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        logfile.disable();

        let input = StoredStructure::load(matches.expect_value_of("input"))?;
        let dim_string = matches.expect_value_of("dims");
        let layer_sc_mode = match matches.value_of("primitive").unwrap_or("auto".into()) {
            "auto" => LayerScMode::Auto,
            "input" => LayerScMode::Assign,
            "keep" => LayerScMode::Multiply,
            "none" => LayerScMode::None,
            s => bail!("invalid --layer-scs: {:?}", s),
        };
        let output = matches.expect_value_of("output");

        crate::cmd::run_make_supercell(input, &dim_string, layer_sc_mode, output)
    });
}

// %% CRATES: binary: rsp2-compute-for-phonopy %%
pub fn compute_for_phonopy(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .about("Computes a single force set for phonopy.")
                .args(&[
                    arg!( input=STRUCTURE "Input structure POSCAR created by Phonopy"),
                    arg!(*output [-o][--output]=PATH "Path for output vasprun.xml file."),
                ])
        });
        let matches = app.get_matches();
        let ConfigArgs(config) = de.resolve_args(&matches)?;

        logfile.disable(); // no trial dir

        let ValidatedSettings(settings) = config.deserialize()?;

        let poscar = Load::load(matches.expect_value_of("input"))?;

        let force = crate::cmd::run_single_force_computation(mpi_on_demand, &settings, poscar)?;

        crate::filetypes::FakeVasprun { force }.save(matches.expect_value_of("output"))?;

        Ok(())
    });
}

// %% CRATES: binary: rsp2-dynmat-at-q %%
pub fn dynmat_at_q(bin_name: &str, version: VersionInfo) -> ! {
    use crate::ui::parse_qpoint::parse_qpoint;

    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .about("Computes the dynamical matrix at a qpoint.")
                .args(&[
                    arg!( input=STRUCTURE "Input structure, in rsp2 structure directory format."),
                    arg!(*qpoint [--qpoint]=KPOINT "\
                        space-separated list of 3 numbers (integers, reals, or rationals as X/Y) \
                        describing the location in units of the reciprocal cell.\
                    ").allow_hyphen_values(true),
                    arg!(*output [-o][--output]=PATH "Path for output dynmat.npz file."),
                ])
        });
        let matches = app.get_matches();
        let (ConfigArgs(config), AppendLog(append_log)) = de.resolve_args(&matches)?;
        append_log.start(logfile)?;

        let ValidatedSettings(settings) = config.deserialize()?;

        let qpoint_frac = parse_qpoint(&matches.expect_value_of("qpoint"))?;
        let structure = StoredStructure::load(matches.expect_value_of("input"))?;

        let dynmat = crate::cmd::run_dynmat_at_q(mpi_on_demand, &settings, qpoint_frac, structure)?;

        dynmat.save(matches.expect_value_of("output"))?;

        Ok(())
    });
}

// %% CRATES: binary: rsp2-test-rayon %%
pub fn test_rayon(bin_name: &str, version: VersionInfo) -> ! {
    use rayon::prelude::*;

    wrap_main(version, |logfile, _mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            clap::App::new(bin_name)
                .about("Runs some heavy code in rayon for debugging parallelism.")
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        logfile.disable();

        let total = {
            (0..1_000_000_000_000_000_u64).into_par_iter()
                .map(|_| expensive_function())
                .map(|x| x as u64)
                .sum::<u64>()
        };

        panic!("We finished!? (total: {})", total);
    });
}

#[inline(never)]
fn expensive_function() -> u32 {
    let x = rand::random::<u16>() as u32;
    let m = match rand::random::<u16>() as u32 {
        0 => 1,
        m => m,
    };

    (0..1_000_000)
        .scan(1, |prod, _| {
            *prod *= x;
            *prod %= m;
            Some(*prod)
        })
        .sum::<u32>()
}

// %% CRATES: binary: rsp2-library-paths %%
pub fn print_library_paths(bin_name: &str, _version: VersionInfo) -> ! {
    let app = {
        clap::App::new(bin_name)
            .about("\
                prints the value of LD_LIBRARY_PATH supplied by cargo at compilation time, \
                to assist in running binaries directly from target/.
            ")
    };
    let _ = app.get_matches();

    println!("{}", std::env::var("LD_LIBRARY_PATH").unwrap());
    exit(0);
}

// -------------------------------------------------------------------------------------
