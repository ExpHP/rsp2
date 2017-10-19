extern crate rsp2_array_utils;

extern crate ordered_float;
extern crate slice_of_array;
extern crate itertools;
#[macro_use] extern crate error_chain;
#[cfg(test)] extern crate rand;

pub mod supercell;

pub use lattice::Lattice;
pub use coords::Coords;
pub use structure::{Structure, CoordStructure};

pub use algo::layer::Layer;
pub use algo::layer::assign_layers;

mod coords;
mod structure;
mod lattice;
mod util;
mod algo;
mod symmops;
