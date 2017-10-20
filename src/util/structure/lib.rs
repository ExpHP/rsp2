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
