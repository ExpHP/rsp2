//! Sparse matrices in various formats.

pub mod coo;
pub mod cs;

pub use self::coo::CooMat;
pub use self::cs::{CscMat, CsrMat};
