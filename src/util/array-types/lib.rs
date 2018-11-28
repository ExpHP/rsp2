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

extern crate rsp2_array_utils;
#[cfg(test)]
#[macro_use]
extern crate rsp2_assert_close;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;
extern crate slice_of_array;
extern crate rand;
extern crate num_traits;

#[macro_use] mod macros;

pub use self::types::*;
mod types;

pub use self::conv::*;
mod conv;

pub use self::ops::*;
mod ops;

mod traits;

#[path = "./methods_v.rs"]
pub mod vee;
#[path = "./methods_m.rs"]
pub mod mat;

pub use crate::vee::dot;
pub use crate::mat::inv;
