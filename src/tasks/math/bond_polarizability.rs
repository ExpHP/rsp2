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

//! Computes raman intensities of gamma eigenkets using
//! a bond polarizability model.
//!
//! Adapted from the sp2 code.

use crate::FailResult;
use crate::math::basis::GammaBasis3;
use crate::meta::{Element, Mass};
use enum_map::EnumMap;
use rsp2_array_types::{dot, V3, M33};
use rsp2_structure::bonds::{CartBond, CartBonds};

/// Quick little struct to simulate named arguments
pub struct Input<'a> {
    pub temperature: f64,
    pub ev_frequencies: &'a [f64],
    pub ev_eigenvectors: &'a GammaBasis3,
    pub site_elements: &'a [Element],
    pub site_masses: &'a [Mass],
    pub bonds: &'a CartBonds,
}

impl<'a> Input<'a> {
    pub fn compute_ev_raman_tensors(self) -> FailResult<Vec<RamanTensor>> {
        unimplemented!("call rsp2-bond-polarizability crate")
    }
}
