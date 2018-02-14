
use ::Result;

#[allow(unused)] // rust-lang/rust#45268
use ::traits::{Save, AsPath};
use ::traits::save::{Json, Yaml};
use ::clap;

use ::std::sync::atomic::{AtomicBool, ATOMIC_BOOL_INIT, Ordering};
use ::std::path::{Path, PathBuf};

use ::ui::cfg_merging::ConfigSources;

/// Arguments shared in common by all rsp2 binaries that use the Trial API.
pub struct CommonArgs {
    output_dir: PathBuf,
    config_sources: ConfigSources,
    err_if_existing: bool,
    verbosity: i32,
}

impl CommonArgs {
    /// Used to construct CommonArgs. (this is done in two stages, hence the callback)
    ///
    /// # Usage
    ///
    /// ```rust,ignore
    /// // (the name resolve_common_args is suggested on the basis that the callback
    /// //  not only deserializes from ArgMatches, but it will also resolve relative
    /// //  paths in the common arguments and may possibly even open and read files)
    /// let (app, resolve_common_args) = augment_clap_app(app);
    /// let matches = app.get_matches();
    /// let common_args = resolve_common_args(&matches)?;
    /// ```
    pub fn augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> (clap::App<'a, 'b>, fn(&clap::ArgMatches) -> Result<Self>) {
        let app = app.args(&[
            arg!(*output [-o][--output]=OUTDIR "output directory"),
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
            arg!( verbose_minus [-q][--quiet]... "reduce verbosity level"),
            arg!( verbose_plus [-v][--verbose]... "increase verbosity level"),
            arg!( force [-f][--force] "replace existing output directories"),
        ]);

        // Note: deliberately not exposed directly as a pub static method,
        //       so that you are forced to call the surrounding method first.
        fn resolve_common_args(matches: &clap::ArgMatches) -> Result<CommonArgs> {
            let path = |s: &str| { let s: &Path = s.as_ref(); s.to_owned() };
            Ok(CommonArgs {
                output_dir: path(matches.value_of("output").expect("BUG! (output was required)")),
                config_sources: {
                    ConfigSources::resolve_from_args({
                        matches.values_of("config").expect("BUG! (config was required)")
                    })?
                },
                err_if_existing: !matches.is_present("force"),
                verbosity: {
                    let plus = matches.occurrences_of("verbose_plus") as i32;
                    let minus = matches.occurrences_of("verbose_minus") as i32;
                    plus - minus
                },
            })
        };

        (app, resolve_common_args)
    }
}

pub use self::global::{Trial, TheOnlyGlobalTrial};
mod global {
    use ::ui::logging::GlobalLogger;
    use super::*;

    /// I regret to inform you that this exists.
    ///
    /// It encapsulates parts of initialization and finalization which are common
    /// to most of rsp2's binary shims, some of which affect the global process
    /// environment or use one-time global APIs such as logging.
    ///
    /// It may only be used ONCE over the course of the entire program.
    ///
    /// This exists only because *no other solution has been found* to the issue
    /// of creating the log file inside the output directory (at least, not without
    /// replacing `fern`!). rsp2 also has historically taken the easy way out of
    /// path management by changing the current directory, which would be incredibly
    /// dangerous in a threaded application.
    ///
    /// On the bright side, at least the dangerous properties of rsp2's trial
    /// functions are now self-evident. All things told, rsp2_tasks was *already*
    /// doing all of these horrible things in its trial functions; only now, the
    /// risk can more clearly be seen in the binary shims.
    pub struct TheOnlyGlobalTrial(CommonArgs);

    pub use self::trial::Trial;
    mod trial {
        use super::*;

        /// Token representing the global trial before it is started.
        ///
        /// Only one can ever be instantiated.
        pub struct Trial(());

        impl Trial {
            pub fn new_only() -> Result<Trial> {
                static THE_ONLY_GLOBAL_TRIAL_HAS_BEGUN: AtomicBool = ATOMIC_BOOL_INIT;
                if THE_ONLY_GLOBAL_TRIAL_HAS_BEGUN.swap(true, Ordering::SeqCst) {
                    bail!("main can only run one trial!");
                }
                Ok(Trial(()))
            }
        }
    }

    impl TheOnlyGlobalTrial {
        pub fn from_args(args: CommonArgs) -> Self
        { TheOnlyGlobalTrial(args) }

        pub fn will_now_commence<B, F>(self, f: F) -> Result<B>
        where F: FnOnce(Trial, ::serde_yaml::Value) -> Result<B>,
        {
            use ::rsp2_fs_util as fsx;

            let TheOnlyGlobalTrial(CommonArgs {
                output_dir, config_sources, err_if_existing, verbosity,
            }) = self;

            let token = Trial::new_only()?;

            // Trials always create a fresh output directory, where all files go.
            if !err_if_existing {
                fsx::rm_rf(&output_dir)?;
            }
            fsx::create_dir(&output_dir)?;
            let cwd_guard = ::util::push_dir(output_dir)?;

            // Log file in the output directory.
            GlobalLogger::default()
                .path("rsp2.log")
                .verbosity(verbosity)
                .apply()?;

            // Make some files that detail as much information as possible about how
            // rsp2 was invoked, solely for the user's benefit.
            {
                let args_file: Vec<_> = ::std::env::args().collect();
                Json(args_file).save("input-cli-args.json")?;
            }

            // NOTE: It doesn't feel right for config handling to be a responsibility of
            //       this function, but I always want these files to be saved, so... bleh.
            Yaml(&config_sources).save("input-config-sources.yaml")?;
            let config = config_sources.into_effective_yaml();
            // This file is saved not just for the user's benefit, but also to allow some
            // commands to operate on an existing output directory.
            Yaml(&config).save("settings.yaml")?;

            // let the shim call a `self` function on the trial token.
            let out = f(token, config)?;

            cwd_guard.pop()?;

            Ok(out)
        }
    }
}

pub use self::existing::ExistingTrial;
mod existing {
    use super::*;
    use ::rsp2_tasks_config::YamlRead;

    pub struct ExistingTrial(PathBuf);

    impl ExistingTrial {
        pub fn resolve_from_path(path: &Path) -> Result<Self> {
            let path = ::util::canonicalize(path)?;
            Ok(ExistingTrial(path.to_path_buf()))
        }

        pub fn read_settings<T>(&self) -> Result<T>
        where T: YamlRead,
        {
            use ::rsp2_fs_util as fsx;
            let file = fsx::open(self.0.join("settings.yaml"))?;
            Ok(YamlRead::from_reader(file)?)
        }
    }
}
