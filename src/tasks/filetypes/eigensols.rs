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
use crate::math::basis::{Ket3, Basis3};
use slice_of_array::prelude::*;
use crate::traits::{Load, AsPath, save::Json};

// Conversion factor phonopy uses to scale the eigenvalues to THz angular momentum.
//    = sqrt(eV/amu)/angstrom/(2*pi)/THz
// 1 ev = 23.061 Kcal/mol (NIST)
// 4.80218700177 sqrt((Kcal/mol)/ev) ~= sqrt(23.061)
const SQRT_EIGENVALUE_TO_THZ: f64 = 15.6333043006705 * 4.80218700177;
//    = THz / (c / cm)
const THZ_TO_WAVENUMBER: f64 = 33.3564095198152;
const SQRT_EIGENVALUE_TO_WAVENUMBER: f64 = SQRT_EIGENVALUE_TO_THZ * THZ_TO_WAVENUMBER;

// dumb serializable stub type that I initially wrote on a whim for
// IPC with python for eigsh, and that is now also accepted as an input format
// by one of the binaries (because I needed *something*, even though this
// format probably does not scale well for the large structures that need it)
//
// Each **row** is a column eigenvector. It is expected that they are orthonormal,
// but they need not be complete.

type Eigenvalue = f64;
type Frequency = f64;

#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct Raw(Vec<Eigenvalue>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

pub struct Eigensols {
    pub frequencies: Vec<Frequency>,
    pub eigenvectors: Basis3,
}

impl Raw {
    pub fn into_eigensols(self) -> FailResult<Eigensols> {
        let Raw(vals, (real, imag)) = self;

        if vals.len() != real.len() || vals.len() != imag.len() {
            bail!("mismatched lengths in eigensols file")
        }

        let mut kets = vec![];
        for (real, imag) in zip_eq!(real, imag) {
            let real = real.nest().to_vec();
            let imag = imag.nest().to_vec();
            kets.push(Ket3 { real, imag });
        }
        let frequencies = vals.into_iter().map(eigenvalue_to_frequency).collect();
        let eigenvectors = Basis3(kets);

        Ok(Eigensols { frequencies, eigenvectors })
    }
}

pub fn eigenvalue_to_frequency(val: Eigenvalue) -> Frequency {
    f64::sqrt(f64::abs(val)) * f64::signum(val) * SQRT_EIGENVALUE_TO_WAVENUMBER
}

impl Load for Eigensols {
    fn load(path: impl AsPath) -> FailResult<Self>
    {
        let Json(raw): Json<Raw> = Load::load(path)?;
        raw.into_eigensols()
    }
}
