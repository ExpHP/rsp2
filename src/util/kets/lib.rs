#[macro_use]
extern crate serde_derive;
extern crate serde;
#[macro_use]
extern crate lazy_static;

pub mod basis;
mod complex;

pub use basis::lossless::Basis;
pub use basis::compact::Basis as LossyBasis;
