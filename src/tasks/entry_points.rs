use ::FailResult;

use ::clap;
use ::cmd::trial::{TrialDir, NewTrialDirArgs};
use ::cmd::StructureFileType;
use ::path_abs::{PathDir, PathFile};
use ::std::ffi::OsStr;
use ::ui::logging::init_global_logger;
use ::ui::cfg_merging::ConfigSources;
use ::ui::cli_deserialize::CliDeserialize;
use ::util::ext_traits::{ArgMatchesExt};

fn wrap_result_main<F>(main: F)
where F: FnOnce() -> FailResult<()>,
{
    main().unwrap_or_else(|e| {
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
                    "guess" => None,
                    _ => bail!("invalid setting for --structure-type"),
                }
            } else { None }
        }))
    }
}

impl OptionalFileType {
    pub fn or_guess(self, path: &PathFile) -> StructureFileType {
        self.0.unwrap_or_else(|| {
            match path.extension().and_then(|s| s.to_str()) {
                Some("yaml") => StructureFileType::LayersYaml,
                Some("vasp") => StructureFileType::Poscar,
                _ => StructureFileType::Poscar,
            }
        })
    }
}

// -------------------------------------------------------------------------------------

// %% CRATES: binary: rsp2 %%
pub fn rsp2() {
    use ::cmd::CliArgs;

    impl CliDeserialize for CliArgs {
        fn _augment_clap_app<'a, 'b>(app: ::clap::App<'a, 'b>) -> ::clap::App<'a, 'b> {
            app.args(&[
                arg!( save_bands [--save-bands] "save phonopy directory with bands at gamma"),
            ])
        }

        fn _resolve_args(m: &::clap::ArgMatches) -> FailResult<Self>
        { Ok(CliArgs {
            save_bands: m.is_present("save_bands"),
        })}
    }

    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            app_from_crate!(", ")
                .args(&[
                    arg!( input=STRUCTURE "input file for structure"),
                ])
        });
        let matches = app.get_matches();
        let (dir_args, (filetype, extra_args)) = de.resolve_args(&matches)?;

        let input = PathFile::new(matches.expect_value_of("input"))?;
        let filetype = OptionalFileType::or_guess(filetype, &input);

        let trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.run_relax_with_eigenvectors(&settings, filetype, &input, extra_args)
    });
}

// %% CRATES: binary: rsp2-shear-plot %%
pub fn shear_plot() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-shear-plot")
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
pub fn save_bands_after_the_fact() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-save-bands")
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
pub fn rerun_analysis() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-rerun-analysis")
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
        trial.rerun_ev_analysis(&settings)
    });
}

// %% CRATES: binary: rsp2-bond-test %%
pub fn bond_test() {
    wrap_result_main(|| {
        let _logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-bond-test")
                .args(&[
                    arg!( input=STRUCTURE ""),
                ])
        });
        let matches = app.get_matches();
        let filetype = de.resolve_args(&matches)?;

        let input = PathFile::new(matches.expect_value_of("input"))?;
        let filetype = OptionalFileType::or_guess(filetype, &input);

        let (coords, _, _, _, _) = ::cmd::read_optimizable_structure(None, None, filetype, &input)?;
        let coords = coords.construct(); // reckless

        let bonds = ::math::bonds::FracBonds::from_brute_force_very_dumb(&coords, 1.8)?;
        let bonds = bonds.to_cart_bonds(&coords);
        ::serde_json::to_writer(::std::io::stdout(), &bonds)?;
        println!(); // flush, dammit
        Ok(())
    });
}

// %% CRATES: binary: rsp2-dynmat-test %%
pub fn dynmat_test() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-dynmat-test")
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
