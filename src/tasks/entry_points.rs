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
use crate::traits::Load;
use crate::ui::logging::{init_global_logger, SetGlobalLogfile};
use crate::ui::cfg_merging::ConfigSources;
use crate::ui::cli_deserialize::CliDeserialize;
use crate::util::ext_traits::{ArgMatchesExt};
use crate::filetypes::{StoredStructure, Eigensols};

use ::rsp2_lammps_wrap::LammpsOnDemand;

use ::clap;
use ::path_abs::{PathDir, PathFile, PathAbs};
use ::std::ffi::OsStr;
use ::std::panic::UnwindSafe;
use ::std::process::exit;

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

            let _pp_guard = ::rsp2_python::add_to_python_path()?;

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

// This initializes MPI so it must be done at the very beginning.
//
// The closure runs on only one process.
fn wrap_main_with_lammps_on_demand(continuation: impl UnwindSafe + FnOnce(Option<LammpsOnDemand>)) -> ! {
    #[cfg(feature = "mpi")] {
        let required = ::mpi::Threading::Serialized;
        let (_universe, actual) = {
            ::mpi::initialize_with_threading(required).expect("Could not initialize MPI!")
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
    info!("  rayon: {} thread(s) on the root process", ::rayon::current_num_threads());
    Ok(())
}

fn show_errors(e: ::failure::Error) {
    for cause in e.iter_chain() {
        error!("{}", cause);
    }

    if ::std::env::var_os("RUST_BACKTRACE") == Some(OsStr::new("1").to_owned()) {
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

    if ::std::env::var_os("LAMMPS_POTENTIALS").is_none() {
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
        trial_dir: PathDir::current_dir()?.join(m.expect_value_of("trial_dir")),
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
            let meta = match path.metadata() {
                Ok(meta) => meta,
                Err(_) => return default,
            };

            if meta.is_file() {
                match path.extension().and_then(|s| s.to_str()) {
                    Some("yaml") => StructureFileType::LayersYaml,
                    Some("vasp") => StructureFileType::Poscar,
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
            ::clap::App::new(bin_name)
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

        let settings = trial.read_base_settings()?;
        trial.run_relax_with_eigenvectors(mpi_on_demand, &settings, filetype, &input, stop_after_dynmat)
    });
}

// %% CRATES: binary: rsp2-after-diagonalization %%
pub fn after_diagonalization(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .about("The next step after rsp2-acgsd-and-dynmat and the negative_modes py script")
                .args(&[
                    arg!( dir=DIR "existing trial directory"),
                ])
        });
        let matches = app.get_matches();
        let ConfigOverrideArgs(overrides) = de.resolve_args(&matches)?;

        let dir = PathDir::new(matches.expect_value_of("dir"))?;
        let mut trial = TrialDir::from_existing(&dir)?;

        // Make sure the run is valid before making a logfile
        let iteration = trial.find_iteration_for_ev_chase()?;

        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = match overrides {
            Some(overrides) => {
                let save_path = trial.modified_settings_path(iteration);
                trial.read_modified_settings(overrides, Some(&save_path))?
            },
            None => trial.read_base_settings()?,
        };

        let DidEvChasing(chased) = trial.run_after_diagonalization(
            mpi_on_demand, &settings, iteration,
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
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( input=FORCES_DIR "phonopy forces dir (try --save-bands in main script)"),
                ])
        });
        let matches = app.get_matches();
        let dir_args = de.resolve_args(&matches)?;

        let input = PathDir::new(matches.expect_value_of("input"))?;

        let mut trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_base_settings()?;
        trial.run_energy_surface(mpi_on_demand, &settings, &input)
    });
}

// %% CRATES: binary: rsp2-save-bands-after-the-fact %%
pub fn save_bands_after_the_fact(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( trial_dir=TRIAL_DIR "existing trial directory"),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        let trial = PathDir::new(matches.expect_value_of("trial_dir"))?;
        let mut trial = TrialDir::from_existing(&trial)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_base_settings()?;
        trial.run_save_bands_after_the_fact(mpi_on_demand, &settings)
    });
}

// %% CRATES: binary: rsp2-rerun-analysis %%
pub fn rerun_analysis(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( dir=DIR "existing trial directory, or a structure directory within one"),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        let dir = PathDir::new(matches.expect_value_of("dir"))?;
        let (mut trial, structure) = crate::cmd::resolve_trial_or_structure_path(&dir, "final.structure")?;

        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_base_settings()?;
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
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( dir=DIR "structure in directory format"),
                    arg!( eigensols=EIGENSOLS "eigensolutions file"),
                    arg!(*output [--output][-o]=EIGENSOLS "output directory. Existing files will be clobbered."),
                    arg!( log [--log]=LOGFILE "append to this logfile"),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        let structure = StoredStructure::load(matches.expect_value_of("dir"))?;

        if let Some(path) = matches.value_of("log") {
            logfile.start(PathFile::create(path)?)?; // (NOTE: create does not truncate)
        }

        let (evals, evecs) = {
            let path = PathFile::new(matches.expect_value_of("eigensols"))?;
            let Eigensols { frequencies, eigenvectors } = Load::load(path)?;
            (frequencies, eigenvectors)
        };

        // reminder: does not fail on existing
        let outdir = PathDir::create(matches.expect_value_of("output"))?;

        let analysis = crate::cmd::run_sparse_analysis(structure, &evals, &evecs)?;

        crate::cmd::write_ev_analysis_output_files(&outdir, &analysis)?;
        Ok(())
    });
}

// %% CRATES: binary: rsp2-bond-test %%
pub fn bond_test(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, _mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
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

        let bonds = rsp2_structure::bonds::FracBonds::from_brute_force(&coords, 1.8)?;
        match matches.is_present("cart") {
            true => ::serde_json::to_writer(::std::io::stdout(), &bonds.to_cart_bonds(&coords))?,
            false => ::serde_json::to_writer(::std::io::stdout(), &bonds)?,
        }

        println!(); // flush, dammit
        Ok(())
    });
}

// %% CRATES: binary: rsp2-dynmat-test %%
pub fn dynmat_test(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, _mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( input=PHONOPY_DIR ""),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;
        let input = PathDir::new(matches.expect_value_of("input"))?;

        logfile.disable(); // no trial dir

        crate::cmd::run_dynmat_test(&input)
    });
}

// %% CRATES: binary: rsp2-plot-vdw %%
pub fn plot_vdw(bin_name: &str, version: VersionInfo) -> ! {
    wrap_main(version, |logfile, mpi_on_demand| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
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
            ::clap::App::new(bin_name)
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

// %% CRATES: binary: rsp2-library-paths %%
pub fn print_library_paths(bin_name: &str, _version: VersionInfo) -> ! {
    let app = {
        ::clap::App::new(bin_name)
            .about("\
                prints the value of LD_LIBRARY_PATH supplied by cargo at compilation time, \
                to assist in running binaries directly from target/.
            ")
    };
    let _ = app.get_matches();

    println!("{}", ::std::env::var("LD_LIBRARY_PATH").unwrap());
    exit(0);
}
