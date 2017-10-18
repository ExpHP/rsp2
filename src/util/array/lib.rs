#[cfg(test)]
#[macro_use]
extern crate rsp2_assert_close;
extern crate num_traits;

#[macro_use] mod macros;
mod traits;
mod small_vec;
mod small_mat;
mod dot;

// FIXME actually put thought into the public API layout.

pub mod prelude {
    pub use ::MatrixDeterminantExt;
    pub use ::MatrixInverseExt;
    pub use ::ArrayFoldExt;
}

// TODO expose mat() and inv() free functions instead
pub use ::small_mat::MatrixDeterminantExt;
pub use ::small_mat::MatrixInverseExt;
pub use ::small_vec::ArrayFoldExt;

// Matrix and vector operations on fixed-size array types.
pub use ::traits::{Field,Ring,Semiring};
pub use ::dot::dot;

// Functional operations on arrays.
pub use ::small_vec::vec_from_fn;
pub use ::small_vec::try_vec_from_fn;
pub use ::small_vec::opt_vec_from_fn;
pub use ::small_mat::mat_from_fn;
