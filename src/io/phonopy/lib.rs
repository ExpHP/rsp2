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

#[macro_use] extern crate failure;
#[macro_use] extern crate serde_derive;

pub use self::filetypes::{conf, Conf};
pub use self::filetypes::symmetry_yaml::{self, SymmetryYaml};
pub use self::filetypes::disp_yaml::{self, DispYaml};
pub use self::filetypes::force_sets::{self, ForceSets};

mod filetypes;
pub mod npy;

pub type FailResult<T> = Result<T, failure::Error>;
