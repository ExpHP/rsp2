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

//! Implementation of the z-oriented Kolmogorov Crespi potential,
//! with computation of the Hessian.

use crate::util::switch;
use rsp2_array_types::{V3, M33};

use super::Params;

//------------------------------------------------------------------

#[inline(never)] // ensure visible in profiling output
pub(super) fn compute(
    interpolate: impl FnOnce(f64) -> (f64, f64, f64),
    params: &Params,
    rij: V3,
) -> (f64, V3, M33) {
    // first check if we are too far away to care
    let distsq = rij.sqnorm();
    let cutoff_end = params.cutoff_end();
    if distsq > cutoff_end * cutoff_end {
        return (0.0, V3::zero(), M33::zero());
    }
    dbg!("rsq: {:.9}", distsq);

    let rdata = RData::compute(distsq, rij);
    let RData { dist, dist_d_r, dist_dd_r_r } = rdata;

    // add the terms repulsive
    let (rep, rep_d_r, rep_dd_r_r) = scaled_repulsive_part(params, distsq, rdata, rij);
    let (attractive, attractive_d_r, attractive_dd_r_r) = attractive_part(params, distsq, rij);

    let total = rep + attractive;
    let total_d_r = rep_d_r + attractive_d_r;
    let total_dd_r_r = rep_dd_r_r + attractive_dd_r_r;

    // apply cutoff factor
    let (cutoff, cutoff_d_r, cutoff_dd_r_r);
    {
        let (value, d_dist, dd_dist_dist) = switch(interpolate, (cutoff_end, params.cutoff_begin), dist);
        cutoff = value;
        cutoff_d_r = d_dist * dist_d_r;
        cutoff_dd_r_r = d_dist * dist_dd_r_r + dd_dist_dist * outer(dist_d_r, dist_d_r);
    }

    // lastly, take into account the cutoff function and its derivative
    let value = cutoff * total + params.value_offset();
    let d_r = cutoff_d_r * total + cutoff * total_d_r;
    let dd_r_r = cutoff_dd_r_r * total + cutoff * total_dd_r_r + plus_t(outer(cutoff_d_r, total_d_r));
    (value, d_r, dd_r_r)
}

/// The attractive part of the potential.
fn attractive_part(params: &Params, distsq: f64, r: V3) -> (f64, V3, M33) {
    debug_assert_eq!(distsq, r.sqnorm());

    // beta = r^2 / z0^2
    let z0sq = params.z0 * params.z0;
    let beta_d_r = 2.0 / z0sq * r;
    let beta_dd_r_r = 2.0 / z0sq; // times the identity matrix

    let beta_inv = z0sq / distsq;

    // -A * beta ** -3
    let value = -params.A * (beta_inv * beta_inv * beta_inv);
    let d_beta = (-3.0) * beta_inv * value;
    let dd_beta_beta = (-4.0) * beta_inv * d_beta;
    dbg!("VA: {:.9}", value);

    let d_r = d_beta * beta_d_r;
    let dd_r_r = d_beta * beta_dd_r_r * M33::eye() + outer(beta_d_r, beta_d_r) * dd_beta_beta;

    (value, d_r, dd_r_r)
}

/// Calculate the square of the transverse distance given the distance squared,
/// the distance vector r.
fn rho_squared(r: V3) -> (f64, V3, M33) {
    let value = r[0]*r[0] + r[1]*r[1];
    let d_r = V3([2.0*r[0], 2.0*r[1], 0.0]);
    let dd_r_r = M33::from_diag(V3([2.0, 2.0, 0.0]));

    (value, d_r, dd_r_r)
}

/// The repulsive function f(rho_ij), note: takes rho _squared_ not rho.
fn crespi_fp(params: &Params, rhosq: f64) -> (f64, f64, f64) {
    // Reformulate in terms of a normalized input:  beta = rho^2 / delta^2
    let beta = rhosq / (params.delta * params.delta);
    let beta_d_rhosq = f64::recip(params.delta * params.delta);

    // f(rho) = exp(-beta)(C_0 + C_1 * beta + C_2 * beta^2)
    let [C0, C2, C4] = params.C2N;
    let poly = C0 + beta * (C2 + beta * C4);
    let poly_d_beta = C2 + 2.0 * beta * C4;
    let poly_dd_beta_beta = 2.0 * C4;
    dbg!("Csum: {:.9}", poly);

    let exp = f64::exp(-beta);
    let exp_d_beta = -exp;
    let exp_dd_beta_beta = exp;
    dbg!("expbeta: {:.9}", exp);

    let value = exp * poly;
    let d_beta = exp_d_beta * poly + exp * poly_d_beta;
    let dd_beta_beta = {
        exp_dd_beta_beta * poly
            + 2.0 * exp_d_beta * poly_d_beta
            + exp * poly_dd_beta_beta
    };
    dbg!("frho: {:.9}", value);

    // Reformulate back as a function of rhosq
    let d_rhosq = d_beta * beta_d_rhosq;
    let dd_rhosq_rhosq = dd_beta_beta * beta_d_rhosq * beta_d_rhosq;
    (value, d_rhosq, dd_rhosq_rhosq)
}

/// The repulsive part of the potential f(rho), but calculates rho from input distance vector
/// r, and unit normal n
fn crespi_f(params: &Params, r: V3) -> (f64, V3, M33) {
    // first we need the transverse distances (aka rho^2)
    let (rhosq, rhosq_d_r, rhosq_dd_r_r) = rho_squared(r);

    // then we calculate f itself
    let (f, f_d_rhosq, f_dd_rhosq_rhosq) = crespi_fp(params, rhosq);

    let value = f;
    let d_r = f_d_rhosq * rhosq_d_r;
    let dd_r_r = f_d_rhosq * rhosq_dd_r_r + f_dd_rhosq_rhosq * outer(rhosq_d_r, rhosq_d_r);

    (value, d_r, dd_r_r)
}

/// `C + fij + fji`
fn unscaled_repulsive_part(params: &Params, distsq: f64, r: V3) -> (f64, V3, M33) {
    debug_assert_eq!(distsq, r.sqnorm());

    // calculate f(rho_ij) for ij and ji
    let (fij, fij_d_r, fij_dd_r_r) = crespi_f(params, r);
    let (fji, fji_d_r, fji_dd_r_r) = crespi_f(params, r);

    let value = params.C + fij + fji;
    let d_r = fij_d_r + fji_d_r;
    let dd_r_r = fij_dd_r_r + fji_dd_r_r;
    dbg!("VR: {:.9}", value);

    (value, d_r, dd_r_r)
}

/// The repulsive part of the potential.
fn scaled_repulsive_part(params: &Params, distsq: f64, rdata: RData, r: V3) -> (f64, V3, M33) {
    // the sum factor
    let RData { dist, dist_d_r, dist_dd_r_r } = rdata;
    let (rep, rep_d_r, rep_dd_r_r) = unscaled_repulsive_part(params, distsq, r);

    // the exponential scaling factor
    let (scale, scale_d_r, scale_dd_r_r);
    {
        scale = f64::exp(-params.lambda * (dist - params.z0));
        let scale_d_dist = -params.lambda * scale;
        let scale_dd_dist_dist = -params.lambda * scale_d_dist;

        // ...in terms of r
        scale_d_r = scale_d_dist * dist_d_r;
        scale_dd_r_r = scale_d_dist * dist_dd_r_r + scale_dd_dist_dist * outer(dist_d_r, dist_d_r);
    }
    dbg!("scale: {:.9}", scale);

    let value = scale * rep;
    let d_r = scale_d_r * rep + scale * rep_d_r;
    let dd_r_r = scale_dd_r_r * rep + scale * rep_dd_r_r + plus_t(outer(scale_d_r, rep_d_r));

    (value, d_r, dd_r_r)
}

struct RData {
    dist: f64,
    dist_d_r: V3,
    dist_dd_r_r: M33,
}

impl RData {
    fn compute(distsq: f64, rij: V3) -> RData {
        debug_assert_eq!(distsq, rij.sqnorm());
        let dist = distsq.sqrt();
        let dist_d_r = rij / dist;
        let dist_dd_r_r = (M33::eye() - outer(dist_d_r, dist_d_r)) / dist;
        RData { dist, dist_d_r, dist_dd_r_r }
    }
}

// computes `a b.T`
#[inline]
fn outer(a: V3, b: V3) -> M33 {
    M33::from_fn(|r, c| a[r] * b[c])
}

fn plus_t(a: M33) -> M33 {
    a + a.t()
}

//------------------------------------------------------------------

#[cfg(test)]
mod numerical_tests {
    use super::*;
    use crate::util::{num_grad_v3};
    use rsp2_minimize::numerical;
    use rsp2_array_types::Unvee;

    // Crank this up to 20000000 to check for overly strict tolerances.
    const NTRIAL: u32 = 20;

    macro_rules! check_numerical_derivatives_33 {
        (
            rel=$rel:expr, abs=$abs:expr,
            step=$step:expr, center=$center:expr,
            $func:expr $(,)?
        ) => {
            let (rel, abs, step, center, func) = ($rel, $abs, $step, $center, $func);
            let (_, expected_grad, expected_hessian): (f64, V3, M33) = func(center);

            assert_close!{
                rel=rel, abs=abs,
                expected_hessian.unvee(),
                expected_hessian.t().unvee(),
            };

            assert_close!{
                rel=rel, abs=abs, expected_grad.0,
                num_grad_v3(step, center, |x| func(x).0).0,
                "grad",
            }

            for k in 0..3 {
                assert_close!{
                    rel=rel, abs=abs, expected_hessian[k].0,
                    num_grad_v3(step, center, |x| func(x).1[k]).0,
                    "hessian",
                }
            }
        };
    }

    #[test]
    fn crespi() {
        let ref params = Params::original();

        for _ in 0..NTRIAL {
            // For the numerically computed hessian to be accurate, we need a 7th degree
            // cutoff polynomial.
            check_numerical_derivatives_33!{
                rel=1e-8, abs=1e-10, step=1e-3, center=params.random_r(),
                |rij| super::compute(switch::raw_poly7, params, rij),
            }

            // However, numerical computation of the hessian is not an expected use case for this
            // potential, since it provides the hessian. So we normally use the 5th degree
            // polynomial, to match the other Kolmogorov/Crespi implementation.
            check_numerical_derivatives_33!{
                // This requires a bigger tolerance
                rel=1e-3, abs=1e-8, step=1e-3, center=params.random_r(),
                |rij| super::compute(switch::raw_poly5, params, rij),
            }
        }
    }

    #[test]
    fn attractive_part() {
        let ref params = Params::original();

        for _ in 0..NTRIAL {
            check_numerical_derivatives_33!{
                rel=1e-11, abs=1e-11, step=1e-3, center=params.random_r(),
                |rij: V3| super::attractive_part(params, rij.sqnorm(), rij),
            }
        }
    }

    #[test]
    fn rho_squared() {
        let ref params = Params::original();

        for _ in 0..NTRIAL {
            check_numerical_derivatives_33!{
                rel=1e-9, abs=1e-9, step=1e-3, center=params.random_r(),
                |r| super::rho_squared(r),
            }
        }
    }

    #[test]
    fn crespi_fp() {
        let ref params = Params::original();

        for _ in 0..NTRIAL {
            let rij = params.random_r();
            let rhosq = V3::dot(&rij, &V3::random_unit());
            let (_, d_rhosq, dd_rhosq_rhosq) = super::crespi_fp(params, rhosq);

            assert_close!{
                rel=1e-10, abs=1e-10, d_rhosq,
                numerical::slope(1e-3, None, rhosq, |rhosq| super::crespi_fp(params, rhosq).0),
            }
            assert_close!{
                rel=1e-10, abs=1e-10, dd_rhosq_rhosq,
                numerical::slope(1e-3, None, rhosq, |rhosq| super::crespi_fp(params, rhosq).1),
            }
        }
    }

    #[test]
    fn crespi_f() {
        let ref params = Params::original();

        for _ in 0..NTRIAL {
            check_numerical_derivatives_33!{
                rel=1e-11, abs=1e-11, step=1e-3, center=params.random_r(),
                |r| super::crespi_f(params, r),
            }
        }
    }

    #[test]
    fn unscaled_repulsive_part() {
        let ref params = Params::original();

        for _ in 0..NTRIAL {
            check_numerical_derivatives_33!{
                rel=1e-11, abs=1e-11, step=1e-3, center=params.random_r(),
                |rij: V3| super::unscaled_repulsive_part(params, rij.sqnorm(), rij),
            }
        }
    }

    #[test]
    fn scaled_repulsive_part() {
        let ref params = Params::original();

        for _ in 0..NTRIAL {
            check_numerical_derivatives_33!{
                rel=1e-10, abs=1e-10, step=1e-3, center=params.random_r(),
                |rij: V3| {
                    let sqnorm = rij.sqnorm();
                    let rdata = RData::compute(sqnorm, rij);
                    super::scaled_repulsive_part(params, sqnorm, rdata, rij)
                },
            }
        }
    }
}
