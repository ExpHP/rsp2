extern crate rsp2_array_utils;

extern crate ordered_float;
extern crate slice_of_array;
extern crate itertools;

pub mod supercell;

pub use lattice::Lattice;
pub use coords::Coords;
pub use structure::{Structure, CoordStructure};

pub use algo::Layer;
pub use algo::assign_layers;

mod coords;
mod structure;
mod lattice;
mod util;
mod algo;
