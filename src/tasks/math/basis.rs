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

use rsp2_kets::{Ket, Basis, AsKetRef};
use rsp2_soa_ops::{Perm, Permute};
use rsp2_soa_ops::{Part, Partition, Unlabeled, helper::partition_each_item};

use crate::hlist_aliases::*;
use crate::meta::{self, Mass, prelude::*};
use rsp2_array_types::V3;
use slice_of_array::prelude::*;

// alternative types to those in rsp2_kets which are defined in terms of 3-vectors
// and for which we can implement Permute and Partition, etc.
/// Orthonormal eigenvectors of the dynamical matrix.
#[derive(Debug, Clone)]
pub struct Basis3(pub(crate) Vec<Ket3>);

#[derive(Debug, Clone)]
pub struct GammaBasis3(pub(crate) Vec<GammaKet3>);

// a ket type belonging exclusively to this crate so that we can say it consists
// of 3-vectors and implement Permute for it, etc
#[derive(Debug, Clone)]
pub struct Ket3 {
    pub(crate) real: Vec<V3>,
    pub(crate) imag: Vec<V3>,
}

#[derive(Debug, Clone)]
pub struct GammaKet3(pub Vec<V3>);

/// Cartesian displacement direction of an eigenvector.
///
/// Not necessarily normalized, and in structures with more than one distinct mass,
/// they may not even be orthogonal.
#[derive(Debug, Clone)]
pub struct EvDirection(Ket3);

impl Basis3 {
    #[allow(unused)]
    pub fn from_basis(basis: Basis) -> Self
    { Basis3(basis.iter().map(|k| Ket3::from_ket(k)).collect()) }

    #[allow(unused)]
    pub fn to_basis(&self) -> Basis {
        let mut basis = Basis::new(vec![], 3 * self.0[0].real.len());
        for ket in &self.0 {
            basis.insert((ket.real.flat(), ket.imag.flat()));
        }
        basis
    }

    pub fn into_gamma_basis3(self) -> Option<GammaBasis3> {
        for ket in self.0.iter() {
            if ket.imag.iter().any(|v| v != &V3::zero()) {
                return None;
            }
        }
        Some(GammaBasis3(self.0.into_iter().map(|ket| GammaKet3(ket.real)).collect()))
    }
}

impl Ket3 {
    #[allow(unused)]
    pub fn from_ket(ket: impl AsKetRef) -> Self
    { Ket3 {
        real: ket.as_ket_ref().real().nest().to_vec(),
        imag: ket.as_ket_ref().imag().nest().to_vec(),
    }}

    pub fn to_ket(&self) -> Ket
    { Ket::new(self.real.flat().to_vec(), self.imag.flat().to_vec()) }
}

impl GammaKet3 {
    #[allow(unused)]
    /// Returns `None` if not real.
    pub fn from_complex(ket: Ket3) -> Option<Self> {
        if ket.imag.iter().any(|v| v != &V3::zero()) {
            return None;
        }
        Some(GammaKet3(ket.real))
    }

    pub fn to_complex(&self) -> Ket3 {
        Ket3 {
            real: self.0.clone(),
            imag: vec![V3::zero(); self.0.len()],
        }
    }

    #[allow(unused)]
    /// Returns `None` if not real.
    pub fn from_ket(ket: impl AsKetRef) -> Option<Self>
    { Self::from_complex(Ket3::from_ket(ket)) }

    pub fn to_ket(&self) -> Ket
    { self.to_complex().to_ket() }
}

impl Permute for Ket3 {
    fn permuted_by(self, perm: &Perm) -> Self
    { Ket3 {
        real: self.real.permuted_by(perm),
        imag: self.imag.permuted_by(perm),
    }}
}

impl Permute for GammaKet3 {
    fn permuted_by(self, perm: &Perm) -> Self
    { GammaKet3(self.0.permuted_by(perm)) }
}

impl Permute for EvDirection {
    fn permuted_by(self, perm: &Perm) -> Self
    { EvDirection(self.0.permuted_by(perm)) }
}

impl<'iter> Partition<'iter> for Ket3 {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {Box::new({
        (self.real, self.imag)
            .into_unlabeled_partitions(part).into_iter()
            .map(|(real, imag)| Ket3 { real, imag })
    })}
}

impl<'iter> Partition<'iter> for GammaKet3 {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {Box::new({
        self.0.into_unlabeled_partitions(part).into_iter().map(GammaKet3)
    })}
}

impl<'iter> Partition<'iter> for Basis3 {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    { Box::new(partition_each_item(part, self.0).map(Basis3)) }
}

impl<'iter> Partition<'iter> for GammaBasis3 {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    { Box::new(partition_each_item(part, self.0).map(GammaBasis3)) }
}

impl<'iter> Partition<'iter> for EvDirection {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    { Box::new(self.0.into_unlabeled_partitions(part).map(EvDirection)) }
}

impl Ket3 {
    #[allow(unused)]
    pub fn sqnorm(&self) -> f64
    {
        self.real.flat().iter().map(|x| x*x).sum::<f64>()
        + self.imag.flat().iter().map(|x| x*x).sum::<f64>()
    }

    pub fn norm(&self) -> f64
    { f64::sqrt(self.sqnorm()) }

    pub fn as_real_checked(&self) -> &[V3]
    {
        for &x in self.imag.flat() {
            assert_eq!(x, 0.0);
        }
        &self.real
    }

    pub fn normalized(&self) -> Self {
        let Ket3 { real, imag } = self;
        let norm = self.norm();
        let real = real.iter().map(|&v| v / norm).collect();
        let imag = imag.iter().map(|&v| v / norm).collect();
        Ket3 { real, imag }
    }
}

impl std::ops::Deref for EvDirection {
    type Target = Ket3;

    fn deref(&self) -> &Ket3
    { &self.0 }
}

// methods that make sense on directions, but not eigenvectors
impl EvDirection {
    pub fn from_eigenvector(evec: &Ket3, meta: HList1<meta::SiteMasses>) -> Self {
        let masses: meta::SiteMasses = meta.pick();
        let (real, imag) = {
            zip_eq!(&evec.real, &evec.imag, &masses[..])
                .map(|(&real, &imag, &Mass(mass)): (&V3, &V3, _)| {
                    (real / f64::sqrt(mass), imag / f64::sqrt(mass))
                })
                .unzip()
        };
        EvDirection(Ket3 { real, imag })
    }

    /// A measure from 0 to `self.sqnorm()` of how acoustic the ket is.
    pub fn acousticness(&self) -> f64
    {
        let square = |x: f64| x * x;

        // dot with an acoustic mode along each axis by simply summing the elements
        // along that axis
        let mut acc = 0.0;
        for k in 0..3 {
            acc += square(self.imag.iter().map(|v| v[k]).sum());
            acc += square(self.real.iter().map(|v| v[k]).sum());
        }
        // factor out the square norm of the acoustic modes
        acc / self.real.len() as f64
    }

    /// Three numbers that sum to `self.sqnorm()` describing which axes the
    /// eigenvector lies along.
    pub fn polarization(&self) -> [f64; 3]
    {
        let mut acc = V3([0f64; 3]);
        for row in ichain!(&self.real, &self.imag,) {
            acc += row.map(|x| x * x);
        }
        acc.0
    }

    pub fn normalized(&self) -> Self
    { EvDirection(self.0.normalized()) }
}
