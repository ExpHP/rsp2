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
#[macro_use] extern crate serde;
#[macro_use] extern crate log;

pub use self::monomorphize::YamlRead;
#[macro_use]
mod monomorphize;

pub mod merge;

#[doc(hidden)] // used by macro
pub mod reexports {
    pub use serde_ignored;
    pub use serde_yaml;
}

pub type FailResult<T> = Result<T, failure::Error>;
