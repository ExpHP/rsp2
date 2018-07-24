/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */
#![cfg_attr(feature = "nightly", feature(euclidean_division))]
#![deny(unused_must_use)]

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
extern crate serde;
#[macro_use] extern crate serde_derive;
#[cfg(test)] extern crate rand;
#[cfg(test)] extern crate serde_json;

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
pub use ::core::structure::Coords;
pub use ::core::structure::NonEquivalentLattice;

pub use ::element::Element;

// yuck. would rather not expose this yet
pub use ::oper::symmops::{IntRot, CartOp};

pub use element::consts as consts;
