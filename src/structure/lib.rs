#![cfg_attr(feature = "nightly", euclidean_division)]

extern crate rsp2_array_utils;
extern crate rsp2_array_types;
extern crate rsp2_soa_ops;
#[macro_use] extern crate rsp2_assert_close;


extern crate ordered_float;
extern crate slice_of_array;
#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate failure;
#[macro_use] extern crate lazy_static;
#[cfg(test)] extern crate rand;

// FIXME copied from failure 1.0 prerelease; remove once actually released
macro_rules! throw {
    ($e:expr) => {
        return Err(::std::convert::Into::into($e));
    }
}

#[derive(Debug, Fail)]
#[fail(display = "Not nearly an integer: {}", value)]
pub struct IntPrecisionError {
    backtrace: ::failure::Backtrace,
    value: f64,
}

pub mod supercell {
    pub use ::algo::supercell::{
        diagonal,
        centered_diagonal,
        Builder,
        OwnedMetas,
        SupercellToken,
        BigDisplacement,
    };
}

pub use ::algo::find_perm;

pub use ::algo::layer;

mod core;
mod algo;
mod oper;
mod util;
mod element;

//---------------------------
// public reexports; API

pub use ::core::lattice::Lattice;
pub use ::core::coords::CoordsKind;
pub use ::core::structure::{Structure, Coords, ElementStructure};
pub use ::core::structure::NonEquivalentLattice;

pub use ::element::Element;

// yuck. would rather not expose this yet
pub use ::oper::symmops::{FracRot, FracTrans, FracOp};

pub use ::algo::find_perm::dumb_symmetry_test;

pub use element::consts as consts;
