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

#[macro_use] extern crate include_dir;
#[macro_use] extern crate log;
use rsp2_fs_util as fsx;

use crate::fsx::TempDir;
use include_dir::{Dir, File};
use std::{
    path::{Path, PathBuf},
    ffi::{OsStr},
    env,
};

pub static ROOT_DIR: Dir<'_> = include_dir!("rsp2");

pub struct Guard(TempDir);

/// Writes rsp2 python modules to a temporary directory and adds it to the `PYTHONPATH` environment
/// variable until the returned RAII guard is dropped.
pub fn add_to_python_path() -> Result<Guard, failure::Error> {
    let temp_dir = TempDir::new_labeled("rsp2", "python package source")?;
    let py_package_dir = temp_dir.path().join("rsp2");

    crate::fsx::create_dir(&py_package_dir)?;
    write_dir_contents(&ROOT_DIR, &py_package_dir)?;

    modify_path_env(
        "PYTHONPATH",
        |paths| paths.insert(0, temp_dir.path().into()),
    )?;
    Ok(Guard(temp_dir))
}

impl Drop for Guard {
    fn drop(&mut self) {
        let result = modify_path_env(
            "PYTHONPATH",
            |paths| {
                // Remove the first match.  Since it's a randomized temp directory that we
                // created in the first place, we shouldn't have to worry that much about
                // subtleties such as what happens if the path appears multiple times.
                if let Some(pos) = paths.iter().position(|p| p == self.0.path()) {
                    paths.remove(pos);
                } else {
                    // clearly, somebody must have modified PYTHONPATH,
                    // but there's no use fretting over it.
                }
            }
        );
        match result {
            Ok(_) => {},
            Err(e) => {
                // NOTE: AFAICT, the only possible Error in `modify_path_env` is when a path
                // contains a colon after `func` ends, which is impossible considering that we
                // didn't add anything new to the env var just now.
                //
                // ...but because it isn't that important, I don't feel brave enough to take the
                // risk of panicking here.  We'll just warn.
                warn!("Error while cleaning up PYTHONPATH modifications: {}", e);
            },
        }
    }
}

fn modify_path_env(
    key: impl AsRef<OsStr>,
    func: impl FnOnce(&mut Vec<PathBuf>),
) -> Result<(), failure::Error> {
    let key = key.as_ref();
    let mut paths = match env::var_os(key) {
        Some(s) => env::split_paths(&s).collect(),
        None => vec![],
    };
    func(&mut paths);
    match paths.len() {
        0 => env::remove_var(key),
        _ => env::set_var(key, env::join_paths(paths)?),
    }
    Ok(())
}

fn write_dir_contents(dir: &Dir<'_>, dest: &Path) -> Result<(), failure::Error> {
    for file in dir.files() {
        write_file(file, &dest.join(file.path().file_name().unwrap()))?;
    }
    for subdir in dir.dirs() {
        let subdest = dest.join(subdir.path().file_name().unwrap());
        fsx::create_dir(&subdest)?;
        write_dir_contents(subdir, &subdest)?;
    }
    Ok(())
}

fn write_file(file: &File<'_>, dest: &Path) -> Result<(), failure::Error> {
    use std::io::Write;

    fsx::create(dest)?.write_all(file.contents())?;
    Ok(())
}
