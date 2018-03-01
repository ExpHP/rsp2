use ::errors::{Result};

use ::clap;
use ::cmd::trial::{TrialDir, NewTrialDirArgs};
use ::path_abs::{PathDir, PathFile};
use ::ui::logging::init_global_logger;
use ::ui::cfg_merging::ConfigSources;
use ::ui::cli_deserialize::CliDeserialize;
use ::util::ArgMatchesExt;

fn wrap_result_main<F>(main: F)
where F: FnOnce() -> Result<()>
{ main().unwrap_or_else(|e| panic!("{}", e.display_chain())); }

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
                Any string containing ':' is interpreted as a literal (no mechanism is provided \
                for escaping this character in a path).  The part after the colon must be a valid \
                yaml value, and there may optionally be a '.'-separated sequence of string keys \
                before the colon (as sugar for constructing a nested mapping). \
            "),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches) -> Result<Self>
    { Ok(NewTrialDirArgs {
        config_sources: ConfigSources::resolve_from_args(m.expect_values_of("config"))?,
        err_if_existing: !m.is_present("force"),
        // FIXME factor out 'absolute()'
        trial_dir: PathDir::current_dir()?.join(m.expect_value_of("trial_dir")),
    })}
}

// -------------------------------------------------------------------------------------

pub fn rsp2() {
    use ::cmd::CliArgs;

    impl CliDeserialize for CliArgs {
        fn _augment_clap_app<'a, 'b>(app: ::clap::App<'a, 'b>) -> ::clap::App<'a, 'b> {
            app.args(&[
                arg!( save_bands [--save-bands] "save phonopy directory with bands at gamma"),
            ])
        }

        fn _resolve_args(m: &::clap::ArgMatches) -> Result<Self>
        { Ok(CliArgs {
            save_bands: m.is_present("save_bands"),
        })}
    }

    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            app_from_crate!(", ")
                .args(&[
                    arg!( input=POSCAR "POSCAR"),
                ])
        });
        let matches = app.get_matches();
        let (dir_args, extra_args) = de.resolve_args(&matches)?;

        let input = PathFile::new(matches.expect_value_of("input"))?;

        let trial = TrialDir::create_new(dir_args)?;
        logfile.start(PathFile::new(trial.new_logfile_path()?)?)?;

        let settings = trial.read_settings()?;
        trial.run_relax_with_eigenvectors(&settings, &input, extra_args)
    });
}

pub fn shear_plot() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-shear-plot")
                .version("negative 0.00.3-734.bubbles")
                .author(crate_authors!{", "})
                .about("blah")
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
        // HACK: Incomplete refactoring; canonicalize() and CanonicalPath should
        //       no longer exist, and this function should take TrialDir
        trial.run_energy_surface(&settings, &input)
    });
}

pub fn save_bands_after_the_fact() {
    wrap_result_main(|| {
        let logfile = init_global_logger()?;

        let (app, de) = CliDeserialize::augment_clap_app({
            ::clap::App::new("rsp2-shear-plot")
                .version("negative 0.00.3-734.bubbles")
                .author(crate_authors!{", "})
                .about("blah")
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
