#[macro_use]
extern crate sp2_assert_close;
extern crate num_traits;

#[macro_use] mod macros;
mod traits;
mod small_vec;
mod small_mat;
mod dot;

// FIXME actually put thought into the public API layout.

pub mod prelude {
    pub use ::slice::prelude::*;
    pub use ::math::prelude::*;
    pub use ::functional::prelude::*;
}

pub mod slice;

pub mod math {
    //! Matrix and vector operations on fixed-size array types.
    pub mod prelude {
        pub use ::small_mat::MatrixDeterminantExt;
        pub use ::small_mat::MatrixInverseExt;
        pub use ::dot::Dot;
    }
    pub use ::traits::{Field,Ring,Semiring};
    pub use ::dot::dot;
}

pub mod functional {
    //! Functional operations on arrays.
    pub mod prelude {
        pub use ::small_vec::ArrayFromFunctionExt;
        pub use ::small_vec::ArrayFoldExt;
    }
    pub use ::small_vec::vec_from_fn;
    pub use ::small_vec::opt_vec_from_fn;
    pub use ::small_vec::try_vec_from_fn;
    pub use ::small_mat::mat_from_fn;
}

pub use math::dot;
pub use functional::vec_from_fn;
pub use functional::mat_from_fn;