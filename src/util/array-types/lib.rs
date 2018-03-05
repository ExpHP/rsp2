
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

// Expose neatly-named modules, but let the .rs files have names that are close alphabetically.
#[doc(hidden)] pub mod methods_v;
#[doc(hidden)] pub mod methods_m;
pub use self::methods_v as vee;
pub use self::methods_m as mat;

pub use self::methods_v::dot;
