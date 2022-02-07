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

use crate::FailResult;
use crate::math::basis::GammaBasis3;
use crate::meta::{Element, Mass};

use rsp2_structure::bonds::{CartBonds};
use rsp2_bond_polarizability as imp;  // implementation moved out to separate crate

pub use imp::{RamanTensor, LightPolarization, Settings};

pub struct Input<'a> {
    pub temperature: f64,
    pub ev_frequencies: &'a [f64],
    pub ev_eigenvectors: &'a GammaBasis3,
    pub site_elements: &'a [Element],
    pub site_masses: &'a [Mass],
    pub bonds: &'a CartBonds,
    pub settings: &'a Settings,
}

impl<'a> Input<'a> {
    pub fn compute_ev_raman_tensors(self) -> FailResult<Vec<RamanTensor>> {
        let site_masses = self.site_masses.iter().map(|&Mass(m)| m).collect::<Vec<_>>();
        imp::Input {
            temperature: self.temperature,
            ev_frequencies: self.ev_frequencies,
            ev_eigenvectors: self.ev_eigenvectors.0.iter().map(|ev| &ev.0[..]),
            site_elements: self.site_elements,
            site_masses: &site_masses,
            bonds: self.bonds,
            settings: self.settings,
        }.compute_ev_raman_tensors().map_err(Into::into)
    }
}
