extern crate rsp2_array_utils;
extern crate rsp2_array_types;
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

#[derive(Debug, Fail)]
#[fail(display = "Not nearly an integer: {}", value)]
pub struct IntPrecisionError {
    backtrace: ::failure::Backtrace,
    value: f64,
}

pub mod helper {
    pub use ::oper::part::composite_perm_for_part_lifo;
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

mod core;
mod algo;
mod oper;
mod util;
mod element;

//---------------------------
// public reexports; API

pub use ::oper::perm::{Perm, Permute};
pub use ::oper::perm::InvalidPermutationError;
pub use ::oper::part::{Part, Parted, Partition, Unlabeled};
pub use ::oper::part::InvalidPartitionError;
pub use ::core::lattice::Lattice;
pub use ::core::coords::CoordsKind;
pub use ::core::structure::{Structure, CoordStructure, ElementStructure};
pub use ::core::structure::NonEquivalentLattice;

pub use ::element::Element;

pub use algo::layer::{find_layers, Layers, LayersPerUnitCell};

// yuck. would rather not expose this yet
pub use ::oper::symmops::{FracRot, FracTrans, FracOp};

pub use ::algo::find_perm::dumb_symmetry_test;

pub use element::consts as consts;

