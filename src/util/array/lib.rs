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
mod types;
pub mod _rsp2_array_types_impl {
    pub use types::*;
}

// FIXME actually put thought into the public API layout.

// Matrix and vector operations on fixed-size array types.
#[deprecated = "vector math should use rsp2-array-types"] pub use ::small_mat::det;
#[deprecated = "vector math should use rsp2-array-types"] pub use ::small_mat::inv;
#[deprecated = "vector math should use rsp2-array-types"] pub use ::dot::dot;

#[deprecated = "this is pointless, use an iterator. (this crate even has a move iter if you really need it)"]
pub use ::small_arr::ArrayFoldExt;

pub use ::traits::{Field, Ring, Semiring};

// Functional operations on arrays.
pub use ::small_arr::{map_arr, try_map_arr, opt_map_arr};
pub use ::small_mat::{map_mat, try_map_mat, opt_map_mat};
pub use ::small_arr::{arr_from_fn, try_arr_from_fn, opt_arr_from_fn};
pub use ::small_mat::{mat_from_fn, try_mat_from_fn, opt_mat_from_fn};

// Iterators on arrays
pub use ::iter::ArrayMoveIterExt;
