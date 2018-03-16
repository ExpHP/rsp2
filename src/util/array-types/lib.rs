
extern crate rsp2_array_utils;
#[cfg(test)]
#[macro_use]
extern crate rsp2_assert_close;

extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate slice_of_array;
#[cfg(test)] extern crate rand;

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

pub use vee::dot;
pub use mat::inv;
