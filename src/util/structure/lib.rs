extern crate sp2_array_utils;
extern crate sp2_slice_of_array;
extern crate ordered_float;

pub mod supercell;

pub use lattice::Lattice;
pub use coords::Coords;
pub use structure::{Structure, CoordStructure};

mod coords;
mod structure;
mod lattice;
mod util;
