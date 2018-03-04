#[cfg(test)]
#[macro_use]
extern crate rsp2_assert_close;
#[cfg(test)]
extern crate rand;

// HACK: This crate shouldn't depend on serde (it's for rsp2-array-types)
extern crate serde;
#[macro_use] extern crate serde_derive;

#[macro_use] mod macros;
mod traits;
mod small_arr;
mod small_mat;
mod dot;
mod iter;
#[cfg(test)]
mod test_util;

// HACK: The contents of array-types actually live in this crate because
//       they make use of traits and macros internal to this crate.
mod vee;
pub mod _rsp2_array_types_impl {
    pub use vee::*;
}

// FIXME actually put thought into the public API layout.

pub use ::small_mat::det;
pub use ::small_mat::inv;
pub use ::small_arr::ArrayFoldExt;

// Matrix and vector operations on fixed-size array types.
pub use ::traits::{Field, Ring, Semiring};
pub use ::dot::dot;

// Functional operations on arrays.
pub use ::small_arr::{map_arr, try_map_arr, opt_map_arr};
pub use ::small_mat::{map_mat, try_map_mat, opt_map_mat};
pub use ::small_arr::{arr_from_fn, try_arr_from_fn, opt_arr_from_fn};
pub use ::small_mat::{mat_from_fn, try_mat_from_fn, opt_mat_from_fn};

// Iterators on arrays
pub use ::iter::ArrayMoveIterExt;

pub use ::vee::{V2, V3, V4, Envee, Unvee, envee, unvee};
