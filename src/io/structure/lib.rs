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

#[macro_use] extern crate rsp2_util_macros;

#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate failure;
#[macro_use] extern crate serde_derive;
use vasp_poscar; // FIXME: ????

pub type FailResult<T> = Result<T, failure::Error>;
#[allow(bad_style)]
pub fn FailOk<T>(x: T) -> Result<T, failure::Error> { Ok(x) }

pub use crate::poscar::Poscar;
mod poscar;

pub use crate::xyz::Xyz;
mod xyz;

pub mod layers_yaml;
pub mod assemble;
