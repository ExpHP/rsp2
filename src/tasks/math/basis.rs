use ::rsp2_kets::{Ket, Basis, AsKetRef};
use ::rsp2_structure::{Perm, Permute};
use ::rsp2_structure::{Part, Partition, Unlabeled};

use ::rsp2_array_types::V3;
use ::slice_of_array::prelude::*;

// alternative types to those in rsp2_kets which are defined in terms of 3-vectors
// and for which we can implement Permute and Partition, etc.
#[derive(Debug, Clone)]
pub struct Basis3(pub(crate) Vec<Ket3>);

// a ket type belonging exclusively to this crate so that we can say it consists
// of 3-vectors and implement Permute for it, etc
#[derive(Debug, Clone)]
pub struct Ket3 {
    pub(crate) real: Vec<V3>,
    pub(crate) imag: Vec<V3>,
}

impl Basis3 {
    pub fn from_basis(basis: Basis) -> Self
    { Basis3(basis.iter().map(|k| Ket3::from_ket(k)).collect()) }

    #[allow(unused)]
    pub fn to_basis(&self) -> Basis
    {
        let mut basis = Basis::new(vec![], 3 * self.0[0].real.len());
        for ket in &self.0 {
            basis.insert((ket.real.flat(), ket.imag.flat()));
        }
        basis
    }
}

impl Ket3 {
    pub fn from_ket<K: AsKetRef>(ket: K) -> Self
    { Ket3 {
        real: ket.as_ket_ref().real().nest().to_vec(),
        imag: ket.as_ket_ref().imag().nest().to_vec(),
    }}

    pub fn to_ket(&self) -> Ket
    { Ket::new(self.real.flat().to_vec(), self.imag.flat().to_vec()) }
}

impl Permute for Ket3 {
    fn permuted_by(self, perm: &Perm) -> Self
    { Ket3 {
        real: self.real.permuted_by(perm),
        imag: self.imag.permuted_by(perm),
    }}
}

impl<'iter> Partition<'iter> for Ket3 {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {Box::new({
        (self.real, self.imag)
            .into_unlabeled_partitions(part).into_iter()
            .map(|(real, imag)| Ket3 { real, imag })
    })}
}

impl<'iter> Partition<'iter> for Basis3
{
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {Box::new({
        self.0.into_iter()
            // (over each ket)
            .map(|x| x.into_unlabeled_partitions(part))
            .fold(
                vec![Basis3(vec![]); part.region_keys().len()],
                |mut bases, kets| {
                    // (over each partition)
                    for (basis, ket) in ::util::zip_eq(&mut bases, kets) {
                        basis.0.push(ket);
                    }
                    bases
                }
            ).into_iter()
    })}
}

impl Ket3 {
    #[allow(unused)]
    pub fn sqnorm(&self) -> f64
    {
        self.real.flat().iter().map(|x| x*x).sum::<f64>()
        + self.imag.flat().iter().map(|x| x*x).sum::<f64>()
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

    pub fn as_real_checked(&self) -> &[V3]
    {
        assert!(self.imag.flat().iter().all(|&x| x == 0.0));
        &self.real
    }
}

