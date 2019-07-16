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

#[macro_use] extern crate rsp2_assert_close;

#[cfg(feature = "test-diff")]
#[macro_use] extern crate pretty_assertions;
#[macro_use] extern crate serde_derive;

use rsp2_fs_util as fsx;

use std::path::PathBuf;

#[macro_use]
mod util;

pub mod cli_test;
pub mod filetypes;
pub use self::cli_test::{CheckFile, CliTest};

pub use self::cli_test::Result;

// used to make queries into tests/resources dryer, with quicker error messages on nonexistent paths
pub fn resource(path: &str) -> PathBuf {
    let dir = path_abs::PathDir::new("tests/resources").unwrap_or_else(|e| panic!("{}", e));
    let path = path_abs::PathAbs::new(dir.as_path().join(path)).unwrap_or_else(|e| panic!("{}", e));
    if !path.as_path().exists() {
        panic!("{}: Path does not exist", path.as_path().display())
    }
    path.as_path().to_owned()
}
