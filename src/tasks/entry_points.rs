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

use ::FailResult;

use ::clap;
use ::cmd::trial::{TrialDir, NewTrialDirArgs};
use ::cmd::StructureFileType;
use ::path_abs::{PathDir, PathFile, PathAbs};
use ::std::ffi::OsStr;
use ::traits::Load;
use ::ui::logging::{init_global_logger, SetGlobalLogfile};
use ::ui::cfg_merging::ConfigSources;
use ::filetypes::{StoredStructure, Eigensols};
use ::ui::cli_deserialize::CliDeserialize;
use ::util::ext_traits::{ArgMatchesExt};

fn wrap_result_main<F>(main: F)
where F: FnOnce(SetGlobalLogfile) -> FailResult<()>,
{
    let wrapped = || {
        let logfile = init_global_logger().expect("Could not init logger");
        check_for_deps()?;
        main(logfile)
    };

    wrapped().unwrap_or_else(|e| {
        for cause in e.causes() {
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
        ::std::process::exit(1);
    });
}

impl CliDeserialize for NewTrialDirArgs {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!(*trial_dir [-o][--output]=OUTDIR "output trial directory"),
            arg!( force [-f][--force] "replace existing output directories"),
            arg!(*config [-c][--config]=CONFIG... "\
                config yaml, provided as either a filepath, or as an embedded literal \
                (via syntax described below). \
                When provided multiple times, the configs are merged according to some fairly \
                dumb strategy, with preference to the values supplied in later arguments. \
                \n\n\
                Literals are written as '--config [NESTED_KEY]:VALID_YAML', \
                where NESTED_KEY is an optional '.'-separated sequence of string keys, \
                and the ':' is a literal colon. When provided, NESTED_KEY constructs a nested \
                mapping (so `--config a.b.c:[2]` is equivalent to `--config :{a: {b: {c: [2]}}}`.\
                \n\n\
                Note that detection of filepaths versus literals is based solely \
                on the presence of a colon, and no means of escaping one in a path \
                are currently provided.\
            "),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches) -> FailResult<Self>
    { Ok(NewTrialDirArgs {
        config_sources: ConfigSources::resolve_from_args(m.expect_values_of("config"))?,
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

    fn _resolve_args(m: &clap::ArgMatches) -> FailResult<Self> {
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

fn check_for_deps() -> FailResult<()> {
    ::cmd::python::check_availability()?;

    if ::std::env::var_os("LAMMPS_POTENTIALS").is_none() {
        bail!("rsp2 requires you to set the LAMMPS_POTENTIALS environment variable.");
    }

    Ok(())
}

// -------------------------------------------------------------------------------------

// %% CRATES: binary: rsp2 %%
pub fn rsp2(_bin_name: &str) {
    wrap_result_main(|logfile| {
        let (app, de) = CliDeserialize::augment_clap_app({
            app_from_crate!(", ")
                .args(&[
                    arg!( input=STRUCTURE "input file for structure"),
                ])
        });
        let matches = app.get_matches();
        let (dir_args, filetype) = de.resolve_args(&matches)?;

        let input = PathAbs::new(matches.expect_value_of("input"))?;
        let filetype = OptionalFileType::or_guess(filetype, &input);

        let trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.run_relax_with_eigenvectors(&settings, filetype, &input)
    });
}

// %% CRATES: binary: rsp2-shear-plot %%
pub fn shear_plot(bin_name: &str) {
    wrap_result_main(|logfile| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( input=FORCES_DIR "phonopy forces dir (try --save-bands in main script)"),
                ])
        });
        let matches = app.get_matches();
        let dir_args = de.resolve_args(&matches)?;

        let input = PathDir::new(matches.expect_value_of("input"))?;

        let trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.run_energy_surface(&settings, &input)
    });
}

// %% CRATES: binary: rsp2-save-bands-after-the-fact %%
pub fn save_bands_after_the_fact(bin_name: &str) {
    wrap_result_main(|logfile| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( trial_dir=TRIAL_DIR "existing trial directory"),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        let trial = PathDir::new(matches.expect_value_of("trial_dir"))?;
        let trial = TrialDir::from_existing(&trial)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.run_save_bands_after_the_fact(&settings)
    });
}

// %% CRATES: binary: rsp2-rerun-analysis %%
pub fn rerun_analysis(bin_name: &str) {
    wrap_result_main(|logfile| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( dir=DIR "existing trial directory, or a structure directory within one"),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;

        let dir = PathDir::new(matches.expect_value_of("dir"))?;
        let (trial, structure) = ::cmd::resolve_trial_or_structure_path(&dir, "final.structure")?;

        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.rerun_ev_analysis(&settings, structure)
    });
}

// FIXME it is kind of dumb having both this and rerun-analysis.
//       the trouble is they take different input; rerun-analysis rediagonalizes the entire
//       system using phonopy (and thus needs access to settings to know the potential),
//       while this requires the eigensolutions as input.
// %% CRATES: binary: rsp2-sparse-analysis %%
pub fn sparse_analysis(bin_name: &str) {

    wrap_result_main(|logfile| {
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

        let analysis = ::cmd::run_sparse_analysis(structure, &evals, &evecs)?;

        ::cmd::write_ev_analysis_output_files(&outdir, &analysis)?;
        Ok(())
    });
}

// %% CRATES: binary: rsp2-bond-test %%
pub fn bond_test(bin_name: &str) {
    wrap_result_main(|_logfile| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( input=STRUCTURE ""),
                    arg!( cart [--cart] "output CartBonds instead of FracBonds")
                ])
        });
        let matches = app.get_matches();
        let filetype = de.resolve_args(&matches)?;

        let input = PathAbs::new(matches.expect_value_of("input"))?;
        let filetype = OptionalFileType::or_guess(filetype, &input);

        let (coords, _, _, _, _) = ::cmd::read_optimizable_structure(None, None, filetype, &input)?;
        let coords = coords.construct(); // reckless

        let bonds = ::math::bonds::FracBonds::from_brute_force_very_dumb(&coords, 1.8)?;
        match matches.is_present("cart") {
            true => ::serde_json::to_writer(::std::io::stdout(), &bonds.to_cart_bonds(&coords))?,
            false => ::serde_json::to_writer(::std::io::stdout(), &bonds)?,
        }

        println!(); // flush, dammit
        Ok(())
    });
}

// %% CRATES: binary: rsp2-dynmat-test %%
pub fn dynmat_test(bin_name: &str) {
    wrap_result_main(|logfile| {
        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new(bin_name)
                .args(&[
                    arg!( input=PHONOPY_DIR ""),
                ])
        });
        let matches = app.get_matches();
        let () = de.resolve_args(&matches)?;
        let input = PathDir::new(matches.expect_value_of("input"))?;

        let _ = logfile; // no trial dir

        ::cmd::run_dynmat_test(&input)
    });
}
