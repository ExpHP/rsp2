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
use rsp2_array_types::V3;

/// Constants used for calculation of the Kolmogorov-Crespi potential.
///
/// These match the values used by default by the implementation of kolmogorov/crespi/z in Lammps,
/// which is scaled to Lammps' rebo's bond length.
pub mod consts {
    /// Transverse distance scaling factor. Units are Angstroms.
    pub const DELTA: f64 = 0.578;

    /// Distance scaling factor for the repulsive potential. Units are inverse Angstroms.
    pub const LAMBDA: f64 = 3.629;

    /// Multiplicative factor for the attractive potential. Units are eV.
    pub const A: f64 = 10.238e-3;

    /// "Convenience scaling factor" Units are Angstroms.
    pub const Z_0: f64 = 3.34;

    /// Offset constant used in repulsive potential. Units are eV.
    pub const C: f64 = 3.030e-3;

    /// Multiplicative constants used in repulsive potential. Contains C_0, C_2, C_4. Units are eV.
    pub const C2N: [f64; 3] = [15.71e-3, 12.29e-3, 4.933e-3];
}

//------------------------------------------------------------------

pub struct Params {
    /// Distance at which the Kolmogorov-Crespi potential starts to switch to zero.
    /// Units are Angstroms.
    pub cutoff_begin: f64,
    /// How far it takes for the Kolmogorov-Crespi to switch to zero after the cutoff distance.
    /// Units are Angstroms.
    pub cutoff_transition_dist: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            cutoff_begin: 11.0,
            cutoff_transition_dist: 2.0,
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
    pub fn compute(&self, r_ij: V3, normal_i: V3, normal_j: V3) -> Output {
        // NOTE: These are debug-only in the hope of optimizing branch prediction
        //       for the cutoff check
        debug_assert_close!(1.0, normal_i.sqnorm());
        debug_assert_close!(1.0, normal_j.sqnorm());
        debug_assert!(self.cutoff_begin >= 0.0);
        debug_assert!(self.cutoff_transition_dist >= 0.0);

        compute(self, r_ij, normal_i, normal_j)
    }

    /// Computes crespi assuming normals are all +Z.
    pub fn crespi_z(&self, r_ij: V3) -> (f64, V3) {
        let z_hat = V3([0.0, 0.0, 1.0]);
        let Output {
            value, grad_rij,
            grad_ni: _,
            grad_nj: _,
        } = self.compute(r_ij, z_hat, z_hat);

        (value, grad_rij)
    }
}

impl Params {
    /// Distance after which the Kolmogorov-Crespi potential is always zero.
    pub fn cutoff_end(&self) -> f64 {
        debug_assert!(self.cutoff_transition_dist >= 0.0);
        self.cutoff_begin + self.cutoff_transition_dist
    }

    /// Produce a randomly oriented vector whose magnitude is most likely
    /// in the transition interval, but also has a reasonably good chance
    /// of being completely inside or completely outside.
    #[cfg(test)]
    fn random_r(&self) -> V3 {
        use ::util::uniform;

        match ::rand::random::<f64>() {
            p if p < 0.10 => (self.cutoff_end() + uniform(0.0, 3.0)) * V3::random_unit(),
            p if p < 0.40 => uniform(3.0, self.cutoff_begin) * V3::random_unit(),
            _ => uniform(self.cutoff_begin, self.cutoff_end()) * V3::random_unit(),
        }
    }
}

//------------------------------------------------------------------

#[inline(never)] // ensure visible in profiling output
fn compute(params: &Params, rij: V3, ni: V3, nj: V3) -> Output {
    // first check if we are too far away to care
    let distsq = rij.sqnorm();
    let cutoff_end = params.cutoff_begin + params.cutoff_transition_dist;
    if distsq > cutoff_end * cutoff_end {
        return Output::zero();
    }

    let dist = distsq.sqrt();
    let dist_d_rij = rij / dist;

    // calculate the value of the multiplicative cutoff parameter as well as its derivative
    // w/r/t distance if we are within the region where the potential switches to zero
    let (cutoff, cutoff_d_dist) = switch::poly5((cutoff_end, params.cutoff_begin), dist);

    // first get the attractive part of the potential
    let (attractive, attractive_d_r) = attractive_part(distsq, rij);

    // then the repulsive
    let (rep, rep_d_r, rep_d_ni, rep_d_nj) = repulsive_part(distsq, rij, ni, nj);

    // as well as its scaling term
    let (scale, scale_d_dist) = crespi_scaling(dist);

    // chain rule, etc, etc, etc
    let value = scale * rep + attractive;
    let d_r = scale_d_dist * dist_d_rij * rep + scale * rep_d_r + attractive_d_r;
    let d_ni = scale * rep_d_ni;
    let d_nj = scale * rep_d_nj;

    // lastly, take into account the cutoff function and its derivative
    Output {
        value: cutoff * value,
        grad_rij: cutoff * d_r + cutoff_d_dist * dist_d_rij * value,
        grad_ni: cutoff * d_ni,
        grad_nj: cutoff * d_nj,
    }
}

/// The attractive part of the potential.
fn attractive_part(distsq: f64, r: V3) -> (f64, V3) {
    debug_assert_eq!(distsq, r.sqnorm());

    use self::consts::{Z_0, A};
    let z0_div_distsq = (Z_0 * Z_0) / distsq;

    // A * (z_0 / rho)^6
    let value = -A * (z0_div_distsq * z0_div_distsq * z0_div_distsq);
    let d_r = -6.0 * r * value / distsq;

    (value, d_r)
}

/// The exponential scaling part of the repulsive potential
fn crespi_scaling(dist: f64) -> (f64, f64) {
    use self::consts::{LAMBDA, Z_0};
    let value = f64::exp(-LAMBDA * (dist - Z_0));
    let d_dist = -LAMBDA * value;

    (value, d_dist)
}

/// Calculate the square of the transverse distance given the distance squared,
/// the distance vector r, and one unit normal vector n
fn rho_squared(distsq: f64, r: V3, n: V3) -> (f64, V3, V3) {
    debug_assert_eq!(distsq, r.sqnorm());
//    debug_assert_close!(1.0, n.sqnorm()); // untrue during numerical differentiation

    let dot = V3::dot(&r, &n);

    // rho^2 = ||r||^2 - (r dot n)^2
    let value = distsq - (dot * dot);
    let d_r = 2.0 * (r - n * dot);
    let d_n = -2.0 * r * dot;

    (value, d_r, d_n)
}

/// The repulsive function f(rho_ij), note: takes rho _squared_ not rho.
fn crespi_fp(rhosq: f64) -> (f64, f64) {
    use self::consts::{DELTA, C2N};

    // Reformulate in terms of a normalized input:  beta = rho^2 / delta^2
    let beta = rhosq / (DELTA * DELTA);
    let beta_d_rhosq = f64::recip(DELTA * DELTA);

    // f(rho) = exp(-beta)(C_0 + C_1 * beta + C_2 * beta^2)
    let poly = C2N[0] + beta * (C2N[1] + beta * C2N[2]);
    let poly_d_beta = C2N[1] + 2.0 * beta * C2N[2];

    let exp = f64::exp(-beta);
    let exp_d_beta = -exp;

    let value = exp * poly;
    let d_beta = exp_d_beta * poly + exp * poly_d_beta;

    // Reformulate back as a function of rhosq
    let d_rhosq = d_beta * beta_d_rhosq;
    (value, d_rhosq)
}

/// The repulsive part of the potential f(rho), but calculates rho from input distance vector
/// r, and unit normal n
fn crespi_f(distsq: f64, r: V3, n: V3) -> (f64, V3, V3) {
    // first we need the transverse distances (aka rho^2)
    let (rhosq, rhosq_d_r, rhosq_d_n) = rho_squared(distsq, r, n);

    // then we calculate f itself
    let (f, f_d_rhosq) = crespi_fp(rhosq);

    let value = f;
    let d_r = f_d_rhosq * rhosq_d_r;
    let d_n = f_d_rhosq * rhosq_d_n;

    (value, d_r, d_n)
}

/// The repulsive part of the potential.
fn repulsive_part(distsq: f64, r: V3, ni: V3, nj: V3) -> (f64, V3, V3, V3) {
    debug_assert_eq!(distsq, r.sqnorm());
//    debug_assert_close!(1.0, ni.sqnorm());  // untrue during numerical differentiation
//    debug_assert_close!(1.0, nj.sqnorm());  // untrue during numerical differentiation

    // calculate f(rho_ij) for ij and ji
    let (fij, fij_d_r, fij_d_ni) = crespi_f(distsq, r, ni);
    let (fji, fji_d_r, fji_d_nj) = crespi_f(distsq, r, nj);

    let value = consts::C + fij + fji;
    let d_r = fij_d_r + fji_d_r;
    let d_ni = fij_d_ni;
    let d_nj = fji_d_nj;

    (value, d_r, d_ni, d_nj)
}

//------------------------------------------------------------------

#[cfg(test)]
mod numerical_tests {
    use super::*;
    use crate::util::{num_grad_v3};
    use rsp2_minimize::numerical;

    // Crank this up to 20000000 to check for overly strict tolerances.
    const NTRIAL: u32 = 20;

    #[test]
    fn crespi() {
        let ref params = Params::default();

        for _ in 0..NTRIAL {
            let rij = params.random_r();
            let ni = V3::random_unit();
            let nj = V3::random_unit();

            let Output {
                value: _,
                grad_rij, grad_ni, grad_nj,
            } = params.compute(rij, ni, nj);

            assert_close!{
                rel=1e-10, abs=1e-10, grad_rij.0,
                num_grad_v3(1e-3, rij, |rij| super::compute(&params, rij, ni, nj).value).0,
            }

            // Finite differences w.r.t. n seem unreliable when n is in the vicinity of r_hat
            let get_n_tol = |n| match V3::dot(&rij.unit(), &n).abs() {
                x if x > 0.98 => 1e-4,
                x if x > 0.9 => 1e-8,
                _ => 1e-10,
            };

            let tol = get_n_tol(ni);
            assert_close!{
                rel=tol, abs=tol, grad_ni.0,
                num_grad_v3(1e-5, ni, |ni| super::compute(&params, rij, ni, nj).value).0,
            }

            let tol = get_n_tol(nj);
            assert_close!{
                rel=tol, abs=tol, grad_nj.0,
                num_grad_v3(1e-5, nj, |nj| super::compute(&params, rij, ni, nj).value).0,
            }
        }
    }

    #[test]
    fn attractive_part() {
        // super::attractive_part(dist_sq: f64, r: V3)
        let ref params = Params::default();

        for _ in 0..NTRIAL {
            let rij = params.random_r();
            let (_, d_rij) = super::attractive_part(rij.sqnorm(), rij);

            assert_close!{
                rel=1e-11, abs=1e-11, d_rij.0,
                num_grad_v3(1e-3, rij, |rij| super::attractive_part(rij.sqnorm(), rij).0).0,
            }
        }
    }

    #[test]
    fn rho_squared() {
        // rho_squared(dist_sq: f64, r: V3, n: V3) -> (f64, V3, V3)
        let ref params = Params::default();
        for _ in 0..NTRIAL {
            let r = params.random_r();
            let n = V3::random_unit();
            let (_, d_r, d_n) = super::rho_squared(r.sqnorm(), r, n);

            assert_close!{
                rel=1e-9, abs=1e-9, d_r.0,
                num_grad_v3(1e-3, r, |r| super::rho_squared(r.sqnorm(), r, n).0).0,
            }
            assert_close!{
                rel=1e-9, abs=1e-9, d_n.0,
                num_grad_v3(1e-3, n, |n| super::rho_squared(r.sqnorm(), r, n).0).0,
            }
        }
    }

    #[test]
    fn crespi_fp() {
        let ref params = Params::default();
        for _ in 0..NTRIAL {
            let rij = params.random_r();
            let rhosq = V3::dot(&rij, &V3::random_unit());
            let (_, d_rhosq) = super::crespi_fp(rhosq);

            assert_close!{
                rel=1e-10, abs=1e-10, d_rhosq,
                numerical::slope(1e-3, None, rhosq, |rhosq| super::crespi_fp(rhosq).0),
            }
        }
    }

    #[test]
    fn crespi_f() {
        let ref params = Params::default();
        for _ in 0..NTRIAL {
            let r = params.random_r();
            let n = V3::random_unit();
            let (_, d_r, d_n) = super::crespi_f(r.sqnorm(), r, n);

            assert_close!{
                rel=1e-11, abs=1e-11, d_r.0,
                num_grad_v3(1e-3, r, |r| super::crespi_f(r.sqnorm(), r, n).0).0,
            }

            // Finite differences w.r.t. n seem unreliable when n is in the vicinity of r_hat
            let n_tol = match V3::dot(&r.unit(), &n).abs() {
                x if x > 0.97 => 1e-7,
                _ => 1e-11,
            };
            assert_close!{
                rel=n_tol, abs=n_tol, d_n.0,
                num_grad_v3(1e-5, n, |n| super::crespi_f(r.sqnorm(), r, n).0).0,
                "unitdot: {}", V3::dot(&r.unit(), &n).abs(),
            }
        }
    }

    #[test]
    fn repulsive_part() {
        let ref params = Params::default();

        for _ in 0..NTRIAL {
            let rij = params.random_r();
            let ni = V3::random_unit();
            let nj = V3::random_unit();

            let (_, d_rij, d_ni, d_nj) = super::repulsive_part(rij.sqnorm(), rij, ni, nj);

            assert_close!{
                rel=1e-11, abs=1e-11, d_rij.0,
                num_grad_v3(1e-3, rij, |rij| super::repulsive_part(rij.sqnorm(), rij, ni, nj).0).0,
            }

            // Finite differences w.r.t. n seem unreliable when n is in the vicinity of r_hat
            let get_n_tol = |n| match V3::dot(&rij.unit(), &n).abs() {
                x if x > 0.97 => 1e-7,
                _ => 1e-11,
            };

            let tol = get_n_tol(ni);
            assert_close!{
                rel=tol, abs=tol, d_ni.0,
                num_grad_v3(1e-5, ni, |ni| super::repulsive_part(rij.sqnorm(), rij, ni, nj).0).0,
                "unitdot: {}", V3::dot(&rij.unit(), &ni).abs(),
            }

            let tol = get_n_tol(nj);
            assert_close!{
                rel=tol, abs=tol, d_nj.0,
                num_grad_v3(1e-5, nj, |nj| super::repulsive_part(rij.sqnorm(), rij, ni, nj).0).0,
                "unitdot: {}", V3::dot(&rij.unit(), &nj).abs(),
            }
        }
    }
}
