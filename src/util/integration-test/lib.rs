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

extern crate failure;
extern crate path_abs;
#[cfg(feature = "test-diff")]
#[macro_use]
extern crate pretty_assertions;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate itertools;

extern crate rsp2_fs_util as fsx;
extern crate rsp2_structure;
extern crate rsp2_python;
extern crate rsp2_array_types;
#[macro_use]
extern crate rsp2_assert_close;

use ::std::path::PathBuf;

#[macro_use]
mod util;

pub mod cli_test;
pub mod filetypes;
pub use self::cli_test::{CheckFile, CliTest};

// used to make queries into tests/resources dryer, with quicker error messages on nonexistent paths
pub fn resource(path: &str) -> PathBuf {
    let dir = ::path_abs::PathDir::new("tests/resources").unwrap_or_else(|e| panic!(e));
    let file = ::path_abs::PathFile::new(dir.join(path)).unwrap_or_else(|e| panic!(e));
    file.into()
}

// used to make queries into tests/resources dryer, with quicker error messages on nonexistent paths
pub fn resource_dir(path: &str) -> PathBuf {
    let dir = ::path_abs::PathDir::new("tests/resources").unwrap_or_else(|e| panic!(e));
    let dir = ::path_abs::PathDir::new(dir.join(path)).unwrap_or_else(|e| panic!(e));
    dir.into()
}
