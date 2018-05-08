use ::rsp2_array_types::{V3, dot};

/// Constants used for calculation of the Kolmogorov-Crespi potential.
mod consts {
    /// Transverse distance scaling factor. Units are Angstroms.
    pub const DELTA: f64 = 0.578_f64;

    /// Distance scaling factor for the repulsive potential. Units are inverse Angstroms.
    pub const LAMBDA: f64 = 3.629_f64;

    /// Multiplicative factor for the attractive potential. Units are eV.
    pub const A: f64 = 10.238e-3;

    /// "Convenience scaling factor" Units are Angstroms.
    pub const Z_0: f64 = 3.34;

    /// Offset constant used in repulsive potential. Units are eV.
    pub const C: f64 = 3.030e-3;

    /// Multiplicative constants used in repulsive potential. Contains C_0, C_2, C_4. Units are eV.
    pub const C2N: [f64; 3] = [15.71e-3, 12.29e-3, 4.933e-3];

    /// Distance at which the Kolmogorov-Crespi potential starts to switch to zero.
    /// Units are Angstroms.
    pub const CUTOFF_BEGIN: f64 = 11.0_f64;

    /// How far it takes for the Kolmogorov-Crespi to switch to zero after the cutoff distance.
    /// Units are Angstroms.
    pub const CUTOFF_TRANSITION_DIST: f64 = 2.0_f64;

    /// Distance after which the Kolmogorov-Crespi potential is always zero.
    pub const CUTOFF_END: f64 = CUTOFF_BEGIN + CUTOFF_TRANSITION_DIST;
}

/// The switching function which takes the potential to zero as the distance passes CUTOFF_BEGIN
fn crespi_switching_func(distance: f64) -> (f64, f64)
{
    debug_assert!(distance >= consts::CUTOFF_BEGIN);
    debug_assert!(distance <= consts::CUTOFF_BEGIN + consts::CUTOFF_TRANSITION_DIST);

    // The switching function S(t) is the Hermite basis function h_00
    // which transitions from 1 -> 0 as t goes from 0 -> 1
    let t = (distance - consts::CUTOFF_BEGIN) / consts::CUTOFF_TRANSITION_DIST;
    (
        // S(t)
        (1.0 + 2.0 * t) * (t - 1.0) * (t - 1.0),
        // d/dt S(t) * dt/dr
        (6.0 * t * (t - 1.0)) / consts::CUTOFF_TRANSITION_DIST
    )
}

/// The attractive part of the potential.
fn crespi_attractive(dist_sq: f64, r: V3) -> (f64, V3)
{
    debug_assert_eq!(dist_sq, r.sqnorm());

    use self::consts::{Z_0, A};
    let z0_div_r_sq = (Z_0 * Z_0) / dist_sq;
    // A * (z_0 / rho)^6
    let value = -A * (z0_div_r_sq * z0_div_r_sq * z0_div_r_sq);
    // gradient w/r/t r
    let grad_r = -6.0 * r * value / dist_sq;

    (value, grad_r)
}

/// The exponential scaling part of the repulsive potential
fn crespi_scaling(dist: f64) -> (f64, f64)
{
    use self::consts::{LAMBDA, Z_0};
    let value = f64::exp(-LAMBDA * (dist - Z_0));
    let d_ddist = -LAMBDA * value;

    (value, d_ddist)
}

/// Calculate the square of the transverse distance given the distance squared,
/// the distance vector r, and one unit normal vector n
fn rho_squared(dist_sq: f64, r: V3, n: V3) -> (f64, V3, V3)
{
    debug_assert_eq!(dist_sq, r.sqnorm());
    debug_assert_close!(1.0, n.sqnorm());

    let nr_dot = dot(&r, &n);

    // rho^2 = ||r||^2 - (r dot n)^2
    let value = dist_sq - (nr_dot * nr_dot);
    // gradient w/r/t r
    let grad_r = 2.0 * (r - n * nr_dot);
    // gradient w/r/t n
    let grad_n = -2.0 * r * nr_dot;

    (value, grad_r, grad_n)
}

/// The repulsive function f(rho_ij), note: takes rho _squared_ not rho.
fn crespi_fp(rho_squared: f64) -> (f64, f64)
{
    use self::consts::{DELTA, C2N};
    // rho^2 / delta^2
    let rho_div_delta_sq = rho_squared / (DELTA * DELTA);

    // f(rho) = exp(-rho^2 / delta^2)) + C_0 + C_1 * (rho^2 / delta^2) + C_2 * (rho^2 / delta^2)^2
    let value = f64::exp(-rho_div_delta_sq) * (
        C2N[0] + rho_div_delta_sq * (C2N[1] + rho_div_delta_sq * C2N[2]));

    // d/d(rho^2) f(rho)
    let d_rho_sq = (-value + f64::exp(-rho_div_delta_sq) * (C2N[1] + rho_div_delta_sq * C2N[2]))
        / (DELTA * DELTA);

    (value, d_rho_sq)
}

/// The repulsive part of the potential f(rho), but calculates rho from input distance vector
/// r, and unit normal n
fn crespi_f(dist_sq: f64, r: V3, n: V3) -> (f64, V3, V3)
{
    // first we need the transverse distances (aka rho^2)
    let (rho_sq, d_rhosq_dr, d_rhosq_dn) = rho_squared(dist_sq, r, n);
    // then we calculate f itself
    let (f, f_d_rhosq) = crespi_fp(rho_sq);

    let value = f;
    let grad_r = f_d_rhosq * d_rhosq_dr;
    let grad_n = f_d_rhosq * d_rhosq_dn;

    (value, grad_r, grad_n)
}

/// The repulsive part of the potential.
fn crespi_repulsive(dist_sq: f64, r: V3, normal_i: V3, normal_j: V3) -> (f64, V3, V3, V3)
{
    debug_assert_eq!(dist_sq, r.sqnorm());
    debug_assert_close!(1.0, normal_i.sqnorm());
    debug_assert_close!(1.0, normal_j.sqnorm());

    // calculate f(rho_ij) for ij and ji
    let (f_ij, f_ij_dr, f_ij_dni) = crespi_f(dist_sq, r, normal_i);
    let (f_ji, f_ji_dr, f_ji_dnj) = crespi_f(dist_sq, r, normal_j);

    let value = consts::C + f_ij + f_ji;
    let grad_r = f_ij_dr + f_ji_dr;
    let grad_ni = f_ij_dni;
    let grad_nj = f_ji_dnj;

    (value, grad_r, grad_ni, grad_nj)
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

/// Calculate the Kolmogorov-Crespi potential given the delta vector between two carbon atoms r_ij
/// as well as the unit normal vectors for each atom, n_i and n_j. Returns the potential as well
/// as the gradient of the potential with respect to r_ij, n_i, and n_j
#[inline(never)] // ensure visible in profiling output
pub fn crespi(r_ij: V3, normal_i: V3, normal_j: V3) -> Output {
    use self::consts::{CUTOFF_BEGIN, CUTOFF_END};
    debug_assert_close!(1.0, normal_i.sqnorm());
    debug_assert_close!(1.0, normal_j.sqnorm());

    // first check if we are too far away to care
    let dist_sq = r_ij.sqnorm();
    if dist_sq > CUTOFF_END * CUTOFF_END {
        // we're too far away, the potential and gradients are always zero
        return Output::zero();
    }

    let dist = dist_sq.sqrt();
    // calculate the value of the multiplicative cutoff parameter as well as its derivative
    // w/r/t distance if we are within the region where the potential switches to zero
    let (cutoff, d_cutoff_ddist) = match dist < CUTOFF_BEGIN {
        true  => (1.0, 0.0),
        false => crespi_switching_func(dist)
    };

    // first get the attractive part of the potential
    let (v_attractive, d_attractive_dr) = crespi_attractive(dist_sq, r_ij);

    // then the repulsive
    let (v_rep, d_rep_dr, d_rep_dni, d_rep_dnj) = crespi_repulsive(dist_sq, r_ij, normal_i, normal_j);

    // as well as its scaling term
    let (v_scale, d_scale_ddist) = crespi_scaling(dist);

    let value = v_scale * v_rep + v_attractive;
    // chain rule, etc, etc, etc
    let grad_r = d_scale_ddist * (r_ij / dist) * v_rep + v_scale * d_rep_dr + d_attractive_dr;
    let grad_ni = v_scale * d_rep_dni;
    let grad_nj = v_scale * d_rep_dnj;

    // lastly, take into account the cutoff function and its derivative
    Output {
        value: cutoff * value,
        grad_rij: cutoff * grad_r + d_cutoff_ddist * (r_ij / dist) * value,
        grad_ni: cutoff * grad_ni,
        grad_nj: cutoff * grad_nj,
    }
}

/// Computes crespi assuming normals are all +Z.
pub fn crespi_z(r_ij: V3) -> Output {
    let z_hat = V3([0.0, 0.0, 1.0]);
    crespi(r_ij, z_hat, z_hat)
}