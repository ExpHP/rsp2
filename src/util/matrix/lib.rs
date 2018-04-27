extern crate rsp2_array_types;
extern crate rsp2_shims;
#[macro_use]
extern crate error_chain;

pub use self::matrix::Matrix;
pub use self::matrix::AsMatrixRefExt;
pub use self::matrix::AsMatrixMutExt;

pub mod prelude {
    use super::AsMatrixRefExt;
    use super::AsMatrixMutExt;
}


mod matrix;
mod linalg;
