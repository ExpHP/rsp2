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

#[macro_use] extern crate rsp2_assert_close;

#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate failure;
#[macro_use] extern crate lazy_static;
#[cfg(feature = "serde")] #[macro_use] extern crate serde;

// FIXME copied from failure 1.0 prerelease; remove once actually released
macro_rules! throw {
    ($e:expr) => {
        return Err(::std::convert::Into::into($e))
    }
}

#[derive(Debug, Fail)]
#[fail(display = "Not nearly an integer: {}", value)]
pub struct IntPrecisionError {
    backtrace: failure::Backtrace,
    value: f64,
}

pub use crate::algo::bonds;
pub use crate::algo::supercell;
pub use crate::algo::find_perm;
pub use crate::algo::layer;

mod core;
mod algo;
mod util;
mod element;
mod symmops;

//---------------------------
// public reexports; API

pub use crate::core::lattice::Lattice;
pub use crate::core::coords::CoordsKind;
pub use crate::core::structure::Coords;
pub use crate::core::structure::NonEquivalentLattice;
pub use crate::algo::find_perm::Missing;
pub use crate::algo::nearest_image::NearestImageFinder;

pub use crate::element::Element;
pub use crate::element::consts as consts;

pub use crate::symmops::{IntRot, CartOp};

pub mod miller {
    //! Utilities for working with Miller indices of planes.
    //!
    //! Generally speaking, RSP2 interprets a Miller index `miller` as a family of planes
    //! defined as the locus of points `x` such that `G · x` is an integer, where `G` is the
    //! reciprocal lattice vector (with no `2*pi` factor) indexed by `miller`.
    //!
    //! This definition naturally supports Miller indices with `gcd != 1`, which represent families
    //! that point in the same direction as the "primitive" Miller index but with `1/n` times the
    //! spacing. (i.e. more densely packed)

    use rsp2_array_types::V3;
    use num_integer::Integer;

    /// Compute the greatest common divisor of a Miller index, as a non-negative integer.
    ///
    /// Calling this with `miller == [0, 0, 0]` is permitted; the result will be `0`.
    pub fn gcd(miller: V3<i32>) -> i32
    { miller[0].gcd(&miller[1]).gcd(&miller[2]) }

    /// Get the "primitive" form of a Miller index, reduced by the GCD.
    ///
    /// Returns `None` for `[0, 0, 0]`.
    pub fn make_primitive(miller: V3<i32>) -> Option<V3<i32>> {
        match gcd(miller) {
            0 => None,
            g => Some(miller.map(|x| x / g)),
        }
    }

    #[cfg(test)]
    pub(crate) fn random_nonzero(max: i32) -> V3<i32> {
        use rand::{thread_rng, Rng};

        assert!(max > 0);
        loop {
            let miller = V3::from_fn(|_| thread_rng().gen_range(-max, max+1));
            if miller == V3::zero() {
                continue;
            }
            return miller;
        }
    }
}
