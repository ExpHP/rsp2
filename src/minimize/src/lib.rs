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

extern crate serde;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate failure;
#[macro_use] extern crate serde_json;

#[cfg_attr(test, macro_use)] extern crate rsp2_assert_close;
#[macro_use] extern crate rsp2_util_macros;
extern crate rsp2_array_utils;
extern crate rsp2_slice_math;

extern crate either;

#[macro_use] extern crate log;
#[cfg_attr(test, macro_use)] extern crate itertools;
extern crate rand;
extern crate objekt;
extern crate ordered_float;
#[cfg(test)] extern crate env_logger;

pub mod test;

pub(crate) mod util;
pub(crate) mod stop_condition;
pub mod acgsd;
pub(crate) mod linesearch;
pub(crate) mod hager_ls;
pub use ::acgsd::cg_descent;
pub use ::hager_ls::linesearch;
pub mod exact_ls;
pub use ::exact_ls::linesearch as exact_ls;
pub mod numerical;
