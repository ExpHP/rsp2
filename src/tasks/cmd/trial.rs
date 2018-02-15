
use ::Result;

#[allow(unused)] // rust-lang/rust#45268
use ::traits::{Save, AsPath};
use ::traits::save::{Json, Yaml};

use ::ui::cfg_merging::ConfigSources;
use ::ui::logging::{GLOBAL_LOGFILE, init_global_logger};
use ::util::{CanonicalPath, canonicalize, canonicalize_parent};
use ::util::{LockfilePath, LockfileGuard};
use ::util::ArgMatchesExt;

use ::clap;

use ::std::sync::atomic::{AtomicBool, ATOMIC_BOOL_INIT, Ordering};
use ::std::path::{Path, PathBuf};

use ::serde_yaml::Value as YamlValue;

use ::rsp2_fs_util as fsx;

pub use self::global::{Trial, TheOnlyGlobalTrial};
mod global {
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
    pub struct TheOnlyGlobalTrial(());

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
        pub fn run_in_new_dir<B, F>(args: NewTrialDirArgs, f: F) -> Result<B>
        where F: FnOnce(Trial, YamlValue) -> Result<B>,
        {
            let NewTrialDirArgs {
                trial_dir, config_sources, err_if_existing,
            } = args;

            let token = Trial::new_only()?;
            let trial_dir = TrialDir::resolve_from_path(&trial_dir)?;

            if !err_if_existing  {
                fsx::rm_rf(&trial_dir)?;
            }
            fsx::create_dir(&trial_dir)?;

            // Obtain a lock before writing anything to the directory.
            let lockfile_guard = {
                match trial_dir.lockfile().try_lock()? {
                    None => bail!("the lockfile was stolen from under our feet!"),
                    Some(g) => g,
                }
            };

            // Make some files that detail as much information as possible about how
            // rsp2 was invoked, solely for the user's benefit.
            {
                let args_file: Vec<_> = ::std::env::args().collect();
                Json(args_file).save(trial_dir.join("input-cli-args.json"))?;
            }

            Yaml(&config_sources).save(trial_dir.join("input-config-sources.yaml"))?;
            let config = config_sources.into_effective_yaml();

            // This file is saved not just for the user's benefit, but also to allow some
            // commands to operate on an existing output directory.
            Yaml(&config).save(trial_dir.settings_path())?;

            TheOnlyGlobalTrial::_run_in_dir(trial_dir, token, &lockfile_guard, f)
        }

        pub fn run_in_existing_dir<B, F>(args: ExistingTrialDirArgs, f: F) -> Result<B>
        where F: FnOnce(Trial, YamlValue) -> Result<B>,
        {
            let ExistingTrialDirArgs { trial_dir } = args;

            let token = Trial::new_only()?;
            let trial_dir = TrialDir::resolve_from_path(&trial_dir)?;

            let lockfile_guard = {
                match trial_dir.lockfile().try_lock()? {
                    None => bail!("could not obtain a lock on the trial directory"),
                    Some(g) => g,
                }
            };

            TheOnlyGlobalTrial::_run_in_dir(trial_dir, token, &lockfile_guard, f)
        }

        fn _run_in_dir<B, F>(
            trial_dir: TrialDir,
            token: Trial,
            _: &LockfileGuard,
            f: F,
        ) -> Result<B>
        where F: FnOnce(Trial, YamlValue) -> Result<B>,
        {
            GLOBAL_LOGFILE.start(trial_dir.new_logfile_path()?)?;

            init_global_logger()?;

            let config = trial_dir.read_settings()?;

            // let the shim call a `self` function on the trial token.
            // let it pretend to live in the trial directory.
            let cwd_guard = ::util::push_dir(trial_dir.path())?;
            let out = f(token, config)?;
            cwd_guard.pop()?;

            Ok(out)
        }
    }
}

pub use self::trial_dir::TrialDir;
mod trial_dir {
    use super::*;
    use ::rsp2_tasks_config::YamlRead;

    pub struct TrialDir(PathBuf);

    impl ::std::ops::Deref for TrialDir {
        type Target = Path;
        fn deref(&self) -> &Path { &self.0 }
    }

    impl AsRef<Path> for TrialDir {
        fn as_ref(&self) -> &Path { &self.0 }
    }

    impl TrialDir {
        pub fn resolve_from_path(path: &Path) -> Result<Self> {
            let path = ::util::canonicalize(path)?;
            Ok(TrialDir(path.to_path_buf()))
        }

        pub fn path(&self) -> &Path
        { &self.0 }

        pub fn lockfile(&self) -> LockfilePath
        { LockfilePath(self.0.join("rsp2.lock")) }

        /// # Errors
        ///
        /// This uses a lockfile (not the same as 'lockfile()'), and will fail if
        /// it cannot be created for some reason other than being locked.
        pub fn new_logfile_path(&self) -> Result<PathBuf>
        {
            use ::rsp2_fs_util as fsx;

            let paths = {
                ::std::iter::once("rsp2.log".into())
                    .chain((0..).map(|i| format!("rsp2.{}.log", i)))
                    .map(|s| self.0.join(s))
            };

            // there *shouldn't* be any other processes making logfiles, but w/e
            let _guard = LockfilePath(self.0.join("logfile-naming-lock")).lock()?;
            for path in paths {
                let path: &Path = path.as_ref();
                if path.exists() { continue }

                // create it now while the lockfile is still held, for atomicity.
                let _ = fsx::create(path)?;

                return Ok(path.to_owned());
            }
            panic!("gee, that's an awful lot of log files you have there");
        }

        pub fn settings_path(&self) -> PathBuf
        { self.0.join("settings.yaml") }

        pub fn read_settings<T>(&self) -> Result<T>
        where T: YamlRead,
        {
            use ::rsp2_fs_util as fsx;
            let file = fsx::open(self.settings_path())?;
            Ok(YamlRead::from_reader(file)?)
        }
    }
}

pub trait CliDeserialize: Sized {
    fn augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> (clap::App<'a, 'b>, ClapDeserializer<Self>)
    {
        let app = Self::_augment_clap_app(app);
        let token = ClapDeserializer(Default::default());
        (app, token)
    }

    /// Don't use this. Call 'augment_clap_app' instead.
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b>;
    /// Don't use this. Call 'resolve_args' on the ClapDeserializer instead.
    fn _resolve_args(matches: &clap::ArgMatches) -> Result<Self>;
}

/// Token of "proof" that a clap app was augmented to be capable of deserializing A.
///
/// (note this requirement can be easily circumvented; it's just a speed bump to
///  catch stupid mistakes)
pub struct ClapDeserializer<A>(::std::marker::PhantomData<A>);

impl<A> ClapDeserializer<A>
where A: CliDeserialize,
{
    /// Deserialize the arguments.  This may perform IO such as eagerly reading input files.
    ///
    /// (that said, implementations of the trait are highly discouraged from doing things
    ///  that would cause the behavior of arg resolution to depend on the order in which
    ///  multiple CliDeserialize instances are handled)
    pub fn resolve_args(self, matches: &clap::ArgMatches) -> Result<A>
    { A::_resolve_args(matches) }
}

// Tuple as product combinator
impl<A, B> CliDeserialize for (A, B)
where
    A: CliDeserialize,
    B: CliDeserialize,
{
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b>
    {
        let app = A::_augment_clap_app(app);
        let app = B::_augment_clap_app(app);
        app
    }

    fn _resolve_args(matches: &clap::ArgMatches) -> Result<Self>
    { Ok((A::_resolve_args(matches)?, B::_resolve_args(matches)?)) }
}

pub struct NewTrialDirArgs {
    trial_dir: Box<CanonicalPath>,
    config_sources: ConfigSources,
    err_if_existing: bool,
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
        trial_dir: canonicalize_parent(m.expect_value_of("trial_dir"))?,
    })}
}

pub struct ExistingTrialDirArgs {
    trial_dir: Box<CanonicalPath>,
}

impl CliDeserialize for ExistingTrialDirArgs {
    fn _augment_clap_app<'a, 'b>(app: clap::App<'a, 'b>) -> clap::App<'a, 'b> {
        app.args(&[
            arg!( trial_dir=TRIAL_DIR "existing trial directory"),
        ])
    }

    fn _resolve_args(m: &clap::ArgMatches) -> Result<Self>
    { Ok(ExistingTrialDirArgs {
        trial_dir: canonicalize(m.expect_value_of("trial_dir"))?,
    })}
}
