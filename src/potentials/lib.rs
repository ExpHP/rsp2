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
#[macro_use] extern crate rsp2_assert_close;
#[macro_use] extern crate rsp2_newtype_indices;
#[macro_use] extern crate rsp2_array_utils;
extern crate rsp2_array_types;
extern crate rsp2_structure;
extern crate rsp2_minimize;
extern crate rsp2_rayon_utils;
#[cfg(test)] extern crate rsp2_structure_io;

#[macro_use] extern crate enum_map;
#[macro_use] extern crate failure;
#[macro_use] extern crate log;
#[macro_use] extern crate lazy_static;
extern crate stack;
extern crate petgraph;
extern crate rayon;
extern crate slice_of_array;
#[cfg(test)] extern crate rand;
#[cfg(test)] extern crate xz2;
#[cfg(test)] extern crate serde;
#[cfg(test)] extern crate serde_json;
#[cfg(test)] #[macro_use] extern crate serde_derive;

pub mod crespi;
pub mod rebo;
pub(crate) mod util;

pub type FailResult<T> = Result<T, ::failure::Error>;
#[allow(bad_style)]
pub fn FailOk<T>(x: T) -> FailResult<T> { Ok(x) }
pub use ::std::io::Result as IoResult;
