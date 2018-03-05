#[macro_use] mod macros;
mod traits;
mod functional;
mod iter;
#[cfg(test)] mod test_util;

// Functional operations on arrays.
pub use ::functional::{map_arr, try_map_arr, opt_map_arr};
pub use ::functional::{arr_from_fn, try_arr_from_fn, opt_arr_from_fn};

// NOTE: There is deliberately no fold operation.  You can iterate over arrays.
//       We even have a move iter!

// Iterators on arrays
pub use ::iter::ArrayMoveIterExt;
