
use ::Result;

use ::traits::{Save, AsPath};
use ::traits::save::{Json, Yaml};
use ::util::{LockfilePath, LockfileGuard};
use ::ui::cfg_merging::ConfigSources;

use ::std::path::{Path};
use ::path_abs::{PathArc, PathDir, PathFile, FileRead, FileWrite};
use ::rsp2_tasks_config::YamlRead;
use ::rsp2_fs_util::rm_rf;

pub struct TrialDir {
    path: PathDir,
    _lock: LockfileGuard,
}

impl AsPath for TrialDir {
    fn as_path(&self) -> &Path { &self.path }
}

pub struct NewTrialDirArgs {
    pub trial_dir: PathArc,
    pub config_sources: ConfigSources,
    pub err_if_existing: bool,
}

impl TrialDir {
    pub fn create_new(args: NewTrialDirArgs) -> Result<TrialDir> {
        let NewTrialDirArgs {
            trial_dir, config_sources, err_if_existing,
        } = args;

        if !err_if_existing {
            rm_rf(&trial_dir)?;
        }
        let trial_dir = PathDir::create(&trial_dir)?;

        // Obtain a lock before writing anything to the directory.
        let trial_dir = TrialDir {
            _lock: match Self::lockfile_path(&trial_dir).try_lock()? {
                None => bail!("the lockfile was stolen from under our feet!"),
                Some(g) => g,
            },
            path: trial_dir,
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
        Yaml(&config).save(Self::_settings_path(&trial_dir))?;

        trial_dir.validate()
    }

    fn lockfile_path(dir: &PathDir) -> LockfilePath
    { LockfilePath(dir.join("rsp2.lock").into()) }

    fn _settings_path(path: &PathDir) -> PathArc
    { PathArc::new(path.join("settings.yaml")) }

    fn settings_path(&self) -> Result<PathFile>
    { Ok(Self::_settings_path(self).canonicalize()?.into_file()?) }

    pub fn from_existing(path: &PathArc) -> Result<Self> {
        let path = PathArc::new(path).canonicalize()?.into_dir()?;
        TrialDir {
            _lock: match Self::lockfile_path(&path).try_lock()? {
                None => bail!("the trial directory is already in use"),
                Some(g) => g,
            },
            path,
        }.validate()
    }

    pub fn validate(self) -> Result<Self> {
        // Double-check that these files exist.
        let _ = self.settings_path()?;
        let _ = Self::lockfile_path(&self).canonicalize()?.into_file()?;
        Ok(self)
    }

    /// # Errors
    ///
    /// This uses a lockfile (not the same as 'lockfile_path()'), and will
    /// fail if it cannot be created for some reason other than being locked.
    pub fn new_logfile_path(&self) -> Result<PathArc>
    {
        let paths = {
            ::std::iter::once("rsp2.log".into())
                .chain((0..).map(|i| format!("rsp2.{}.log", i)))
                .map(|s| self.join(s))
        };

        // there *shouldn't* be any other processes making logfiles, but w/e
        let _guard = LockfilePath(self.join("logfile-naming-lock").into()).lock()?;
        for path in paths {
            if path.exists() { continue }

            // create it now while the lockfile is still held, for atomicity.
            let _ = PathFile::create(&path)?;

            return Ok(path.into());
        }
        panic!("gee, that's an awful lot of log files you have there");
    }

    pub fn create_file<P: AsPath>(&self, path: P) -> Result<FileWrite>
    { Ok(FileWrite::create(self.join(path))?) }

    pub fn open<P: AsPath>(&self, path: P) -> Result<FileRead>
    { Ok(PathFile::new(self.join(path))?.read()?) }

    pub fn read_settings<T>(&self) -> Result<T>
    where T: YamlRead,
    {
        let file = FileRead::read(self.settings_path()?)?;
        Ok(YamlRead::from_reader(file)?)
    }
}

impl ::std::ops::Deref for TrialDir {
    type Target = PathDir;
    fn deref(&self) -> &PathDir { &self.path }
}
