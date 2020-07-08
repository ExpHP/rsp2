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

use crate::util::switch;
use rsp2_array_types::{V3, M33};

//------------------------------------------------------------------

/// Emit tracing debug output.
///
/// Optimizations will constant-fold this so that no code is emitted if the requisite
/// env var is not defined during compilation.
///
/// Ideally, you also have a version of lammps that is patched to produce similar output
/// to the usage of `dbg!` in this file. See the `sorted-diff` and `filter-out` scripts
/// in the rebo test directory for how to compare these outputs.
macro_rules! dbg {
    ($($t:tt)*) => {
        if option_env!("RSP2_CRESPI_TRACE") == Some("1".as_ref()) {
            println!($($t)*);
        }
    };
}

//------------------------------------------------------------------

// Implementations
mod full;
mod z_hessian;

//------------------------------------------------------------------

pub struct Params {
    /// Transverse distance scaling factor. Units are Angstroms.
    pub delta: f64,
    /// Distance scaling factor for the repulsive potential. Units are inverse Angstroms.
    pub lambda: f64,
    /// Multiplicative factor for the attractive potential. Units are eV.
    pub A: f64,
    /// "Convenience scaling factor." Units are Angstroms.
    pub z0: f64,
    /// Offset constant used in repulsive potential. Units are eV.
    pub C: f64,
    /// Multiplicative constants used in repulsive potential.
    /// Contains C0, C2, C4. Units are eV.
    pub C2N: [f64; 3],

    /// Distance at which the Kolmogorov-Crespi potential starts to switch to zero.
    /// Units are Angstroms.
    pub cutoff_begin: f64,
    /// How far it takes for the Kolmogorov-Crespi to switch to zero after the cutoff distance.
    /// Units are Angstroms.
    ///
    /// A value of `None` can be used to simulate Lammps' behavior of a sharp cutoff.
    /// In this case, the value will be offset slightly for C0 continuity.
    pub cutoff_transition_dist: Option<f64>,
}

impl Params {
    /// Constants used for calculation of the Kolmogorov-Crespi potential.
    ///
    /// These match the values used by default by the implementation of `kolmogorov/crespi/z` in
    /// Lammps, which is scaled to Lammps' rebo's bond length.
    ///
    /// # Citation
    /// A.N. Kolmogorov & V. H. Crespi,
    /// Registry-dependent interlayer potential for graphitic systems.
    /// Physical Review B 71, 235415 (2005)
    pub fn original() -> Params {
        let meV = 1e-3;
        Params {
            delta: 0.578, // Angstroms
            lambda: 3.629, // Angstroms
            A: 10.238 * meV, // eV
            z0: 3.34, // Angstroms
            C: 3.030 * meV, // eV
            C2N: [15.71 * meV, 12.29 * meV, 4.933 * meV], // eV
            cutoff_begin: 11.0,
            cutoff_transition_dist: Some(2.0),
        }
    }

    /// Constants used for calculation of the Kolmogorov-Crespi potential.
    ///
    /// These match the values used by default by the implementation of `kolmogorov/crespi/full` in
    /// Lammps.
    ///
    /// # Citation
    /// Wengen Ouyang, Davide Mandelli, Michael Urbakh, Oded Hod, arXiv:1806.09555 (2018).
    pub fn ouyang() -> Params {
        let meV = 1e-3;
        Params {
            delta: 0.7718101, // Angstroms
            lambda: 3.143921, // Angstroms
            A: 12.660270 * meV,
            z0: 3.328819, // Angstroms
            C: 6.678908e-4 * meV,
            C2N: [21.847167 * meV, 12.060173 * meV, 4.711099 * meV],
            cutoff_begin: 11.0,
            cutoff_transition_dist: Some(2.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Output {
    pub value: f64,
    pub grad_rij: V3,
    pub grad_ni: V3,
    pub grad_nj: V3,
}

impl Output {
    fn zero() -> Self {
        Output {
            value: 0.0,
            grad_rij: V3::zero(),
            grad_ni: V3::zero(),
            grad_nj: V3::zero(),
        }
    }
}

impl Params {
    /// Calculate the Kolmogorov-Crespi potential.
    ///
    /// Accepts the delta vector between two carbon atoms `r_ij` as well as the unit normal vectors
    /// for each atom, `n_i` and `n_j`. Returns the potential as well as the gradient of the
    /// potential with respect to `r_ij`, `n_i`, and `n_j`.
    ///
    /// For the total potential of a system, one should only sum over undirected bonds.
    /// (see `FracBond::is_canonical`).
    pub fn compute(&self, r_ij: V3, normal_i: V3, normal_j: V3) -> Output {
        // NOTE: These are debug-only in the hope of optimizing branch prediction
        //       for the cutoff check
        debug_assert_close!(1.0, normal_i.sqnorm());
        debug_assert_close!(1.0, normal_j.sqnorm());
        debug_assert!(self.cutoff_begin >= 0.0);
        debug_assert!(self.cutoff_transition_dist.unwrap_or(0.0) >= 0.0);

        self::full::compute(self, r_ij, normal_i, normal_j)
    }

    /// Computes crespi assuming normals are all +Z.
    pub fn compute_z(&self, r_ij: V3) -> (f64, V3) {
        let z_hat = V3([0.0, 0.0, 1.0]);
        let Output {
            value, grad_rij,
            grad_ni: _,
            grad_nj: _,
        } = self.compute(r_ij, z_hat, z_hat);

        (value, grad_rij)
    }

    /// Computes crespi assuming normals are all +Z.
    pub fn compute_z_with_hessian(&self, r_ij: V3) -> (f64, V3, M33) {
        self::z_hessian::compute(switch::raw_poly5, self, r_ij)
    }
}

impl Params {
    /// Distance after which the Kolmogorov-Crespi potential is always zero.
    pub fn cutoff_end(&self) -> f64 {
        debug_assert!(self.cutoff_transition_dist.unwrap_or(0.0) >= 0.0);
        self.cutoff_begin + self.cutoff_transition_dist.unwrap_or(0.0)
    }

    /// Offset added to value for C0 continuity at the cutoff.
    ///
    /// This is zero if a smooth cutoff is enabled (`cutoff_transition_dist`).
    pub fn value_offset(&self) -> f64 {
        match self.cutoff_transition_dist {
            None => {
                // FIXME: we should precompute this, but then we need to do away with the public
                //        fields in Param's API...
                let r = self.cutoff_end();
                let beta = self.z0 / r;
                let beta3 = beta*beta*beta;
                let beta6 = beta3*beta3;
                self.A * beta6
            },
            Some(_) => 0.0,
        }
    }

    /// Produce a randomly oriented vector whose magnitude is most likely
    /// in the transition interval, but also has a reasonably good chance
    /// of being completely inside or completely outside.
    #[cfg(test)]
    fn random_r(&self) -> V3 {
        use crate::util::uniform;

        match rand::random::<f64>() {
            p if p < 0.10 => (self.cutoff_end() + uniform(0.0, 3.0)) * V3::random_unit(),
            p if p < 0.40 => uniform(3.0, self.cutoff_begin) * V3::random_unit(),
            _ => uniform(self.cutoff_begin, self.cutoff_end()) * V3::random_unit(),
        }
    }
}

// Helps downstream code compute the "local normal" as defined in the Kolmogorov/Crespi paper.
pub use crate::util::geometry::unit_cross;
pub use crate::util::geometry::unit;
