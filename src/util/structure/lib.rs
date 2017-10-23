extern crate rsp2_array_utils;

extern crate ordered_float;
extern crate slice_of_array;
#[macro_use] extern crate itertools;
#[macro_use] extern crate error_chain;
#[macro_use] extern crate lazy_static;
#[cfg(test)] extern crate rand;

error_chain!{
    errors {
        BadPerm {
            description("Tried to construct an invalid permutation.")
            display("Tried to construct an invalid permutation.")
        }
        BigDisplacement(d: f64) {
            description("Suspiciously large movement between supercell images."),
            display("Suspiciously large movement between supercell images: {:e}", d),
        }
        IntPrecisionError(d: f64) {
            description("Poor precision for float approximation of integer."),
            display("Not nearly an integer: {}", d),
        }
    }
}
mod errors {
    pub use ::{Result, Error, ErrorKind, ResultExt};
}

#[cfg(test)]
macro_rules! assert_matches {
    ($pat:pat, $expr:expr,)
    => { assert_matches!($pat, $expr) };
    ($pat:pat, $expr:expr)
    => { assert_matches!($pat, $expr, "actual {:?}", $expr) };
    ($pat:pat, $expr:expr, $($arg:expr),+ $(,)*)
    => {
        match $expr {
            $pat => {},
            _ => panic!(
                "assertion failed: {} ({})",
                stringify!(assert_matches!($pat, $expr)),
                format_args!($($arg),+))
        }
    };
}

pub mod supercell;

pub use lattice::Lattice;
pub use coords::Coords;
pub use element::Element;
pub use structure::{Structure, CoordStructure, ElementStructure};

pub use algo::layer::Layer;
pub use algo::layer::assign_layers;

// yuck. would rather not expose this yet
pub use symmops::{FracRot, FracTrans, FracOp};

mod coords;
mod structure;
mod lattice;
mod util;
mod algo;
mod symmops;
mod element;

pub use algo::perm::dumb_symmetry_test;

pub use element::consts as consts;
