extern crate rsp2_array_utils;

extern crate ordered_float;
extern crate slice_of_array;
extern crate itertools;
#[macro_use] extern crate error_chain;
#[macro_use] extern crate lazy_static;
#[cfg(test)] extern crate rand;

error_chain!{
    errors {
        BadPerm {
            description("Tried to construct an invalid permutation.")
            display("Tried to construct an invalid permutation.")
        }
    }
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

mod coords;
mod structure;
mod lattice;
mod util;
mod algo;
mod symmops;
mod element;

pub use element::consts as consts;
