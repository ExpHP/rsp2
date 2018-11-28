/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

use crate::FailResult;

use crate::traits::{Save, AsPath};
use crate::traits::save::{Json, Yaml};
use crate::util::{LockfilePath, LockfileGuard};
use crate::util::ext_traits::PathNiceExt;
use crate::ui::cfg_merging::ConfigSources;

use std::path::{Path};
use path_abs::{PathArc, PathDir, PathFile, FileRead, FileWrite};
use rsp2_tasks_config::YamlRead;
use rsp2_fs_util::rm_rf;

pub struct TrialDir {
    path: PathDir,
    _lock: LockfileGuard,
    // it is a logic error to read the settings more than once
    settings_were_read: bool,
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
    pub fn create_new(args: NewTrialDirArgs) -> FailResult<TrialDir> {
        let NewTrialDirArgs {
            trial_dir, config_sources, err_if_existing,
        } = args;

        if !err_if_existing {
            rm_rf(&trial_dir)?;
        }

        if trial_dir.exists() {
            // PathDir::create does not error on pre-existing dirs...
            bail!(
                "'{}': Output directory already exists! \
                Use --force if you really want to replace it.",
                trial_dir.nice(),
            )
        }
        let trial_dir = PathDir::create(&trial_dir)?;

        // Obtain a lock before writing anything to the directory.
        let trial_dir = TrialDir {
            settings_were_read: false,
            _lock: match Self::lockfile_path(&trial_dir).try_lock()? {
                None => bail!("the lockfile was stolen from under our feet!"),
                Some(g) => g,
            },
            path: trial_dir,
        };

        // Make some files that detail as much information as possible about how
        // rsp2 was invoked, solely for the user's benefit.
        {
            let args_file: Vec<_> = std::env::args().collect();
            Json(args_file).save(trial_dir.join("input-cli-args.json"))?;
        }

        Yaml(&config_sources).save(trial_dir.join("input-config-sources.yaml"))?;
        let config = config_sources.into_effective_yaml();

        // This file is saved not just for the user's benefit, but also to allow some
        // commands to operate on an existing output directory.
        Yaml(&config).save(Self::_base_settings_path(&trial_dir))?;

        trial_dir.validate()
    }

    fn lockfile_path(dir: &PathDir) -> LockfilePath
    { LockfilePath(dir.join("rsp2.lock").into()) }

    fn _base_settings_path(path: &PathDir) -> PathArc
    { PathArc::new(path.join("settings.yaml")) }

    fn base_settings_path(&self) -> FailResult<PathFile>
    { Ok(Self::_base_settings_path(self).canonicalize()?.into_file()?) }

    pub fn from_existing(path: &PathArc) -> FailResult<Self> {
        let path = PathArc::new(path).canonicalize()?.into_dir()?;
        TrialDir {
            settings_were_read: false,
            _lock: match Self::lockfile_path(&path).try_lock()? {
                None => bail!("the trial directory is already in use"),
                Some(g) => g,
            },
            path,
        }.validate()
    }

    pub fn validate(self) -> FailResult<Self> {
        // Double-check that these files exist.
        let _ = self.base_settings_path()?;
        let _ = Self::lockfile_path(&self).canonicalize()?.into_file()?;
        Ok(self)
    }

    /// # Errors
    ///
    /// This uses a lockfile (not the same as 'lockfile_path()'), and will
    /// fail if it cannot be created for some reason other than being locked.
    pub fn new_logfile_path(&self) -> FailResult<PathArc>
    {
        let paths = {
            std::iter::once("rsp2.log".into())
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

    pub fn create_file(&self, path: impl AsPath) -> FailResult<FileWrite>
    { Ok(FileWrite::create(self.join(path))?) }

    #[allow(unused)]
    pub fn read_file(&self, path: impl AsPath) -> FailResult<FileRead>
    { Ok(PathFile::new(self.join(path))?.read()?) }

    // HACK: A quick check for a simple logic error.
    //       (settings should ONLY be read in the entry_point, with borrows propagated
    //        down throughout the rest of the code. This is because they may be monkey-patched
    //        in secondary runs on the same trial dir)
    fn mark_settings_read(&mut self) {
        if self.settings_were_read {
            panic!("\
                (BUG) A trial dir's settings file was read multiple times! This is likely \
                a logic error.\
            ");
        }
        self.settings_were_read = true;
    }

    pub fn read_base_settings<T>(&mut self) -> FailResult<T>
    where T: YamlRead,
    {
        self.mark_settings_read();
        let file = FileRead::read(self.base_settings_path()?)?;
        Ok(YamlRead::from_reader(file)?)
    }

    pub fn read_modified_settings<T>(
        &mut self,
        mut sources: crate::ui::cfg_merging::ConfigSources,
        save_path: Option<impl AsPath>,
    ) -> FailResult<T>
    where T: YamlRead,
    {
        self.mark_settings_read();

        // Get base settings, monkey-patched with the additional sources
        sources.prepend_file(self.base_settings_path()?)?;

        // Possibly save them
        let yaml = sources.into_effective_yaml();
        if let Some(save_path) = save_path {
            trace!("Patched settings for this run will be recorded at {}", save_path.as_path().nice());
            Yaml(&yaml).save(save_path)?;
        }

        // (better error messages for type errors if we reparse from a string)
        let s = serde_yaml::to_string(&yaml)?;
        YamlRead::from_reader(s.as_bytes())
    }
}

impl std::ops::Deref for TrialDir {
    type Target = PathDir;
    fn deref(&self) -> &PathDir { &self.path }
}
