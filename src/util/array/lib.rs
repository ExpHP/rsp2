#[cfg(test)]
#[macro_use]
extern crate rsp2_assert_close;
extern crate num_traits;

#[macro_use] mod macros;
mod traits;
mod small_vec;
mod small_mat;
mod dot;
mod iter;
#[cfg(test)]
mod test_util;

// FIXME actually put thought into the public API layout.

pub use ::small_mat::det;
pub use ::small_mat::inv;
pub use ::small_vec::ArrayFoldExt;

// Matrix and vector operations on fixed-size array types.
pub use ::traits::{Field, Ring, Semiring};
pub use ::dot::dot;

// Functional operations on arrays.
pub use ::small_vec::{map_arr, try_map_arr, opt_map_arr};
pub use ::small_mat::{map_mat, try_map_mat, opt_map_mat};
pub use ::small_vec::{vec_from_fn, try_vec_from_fn, opt_vec_from_fn};
pub use ::small_mat::{mat_from_fn, try_mat_from_fn, opt_mat_from_fn};

// Iterators on arrays
pub use ::iter::ArrayMoveIterExt;
