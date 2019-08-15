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

#![allow(non_snake_case)]

//! Implementation of the full Kolmogorov Crespi potential.

use crate::util::switch;
use rsp2_array_types::V3;

use super::{Params, Output};

#[inline(never)] // ensure visible in profiling output
pub(super) fn compute(params: &Params, rij: V3, ni: V3, nj: V3) -> Output {
    // first check if we are too far away to care
    let distsq = rij.sqnorm();
    let cutoff_end = params.cutoff_end();
    if distsq > cutoff_end * cutoff_end {
        return Output::zero();
    }
    dbg!("rsq: {:.9}", distsq);

    let dist = distsq.sqrt();
    let dist_d_rij = rij / dist;

    // calculate the value of the multiplicative cutoff parameter as well as its derivative
    // w/r/t distance if we are within the region where the potential switches to zero
    let (cutoff, cutoff_d_dist, _) = switch::poly5((cutoff_end, params.cutoff_begin), dist);

    // first get the attractive part of the potential
    let (attractive, attractive_d_r) = attractive_part(params, distsq, rij);

    // then the repulsive
    let (rep, rep_d_r, rep_d_ni, rep_d_nj) = repulsive_part(params, distsq, rij, ni, nj);

    // as well as its scaling term
    let scale = f64::exp(-params.lambda * (dist - params.z0));
    let scale_d_dist = -params.lambda * scale;
    dbg!("scale: {:.9}", scale);

    // chain rule, etc, etc, etc
    let value = scale * rep + attractive;
    let d_r = scale_d_dist * dist_d_rij * rep + scale * rep_d_r + attractive_d_r;
    let d_ni = scale * rep_d_ni;
    let d_nj = scale * rep_d_nj;

    // lastly, take into account the cutoff function and its derivative
    Output {
        value: cutoff * value + params.value_offset(),
        grad_rij: cutoff * d_r + cutoff_d_dist * dist_d_rij * value,
        grad_ni: cutoff * d_ni,
        grad_nj: cutoff * d_nj,
    }
}

/// The attractive part of the potential.
fn attractive_part(params: &Params, distsq: f64, r: V3) -> (f64, V3) {
    debug_assert_eq!(distsq, r.sqnorm());

    let z0_div_distsq = (params.z0 * params.z0) / distsq;

    // A * (z_0 / rho)^6
    let value = -params.A * (z0_div_distsq * z0_div_distsq * z0_div_distsq);
    let d_r = -6.0 * r * value / distsq;
    dbg!("VA: {:.9}", value);

    (value, d_r)
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
fn crespi_fp(params: &Params, rhosq: f64) -> (f64, f64) {
    // Reformulate in terms of a normalized input:  beta = rho^2 / delta^2
    let beta = rhosq / (params.delta * params.delta);
    let beta_d_rhosq = f64::recip(params.delta * params.delta);

    // f(rho) = exp(-beta)(C_0 + C_1 * beta + C_2 * beta^2)
    let [C0, C2, C4] = params.C2N;
    let poly = C0 + beta * (C2 + beta * C4);
    let poly_d_beta = C2 + 2.0 * beta * C4;
    dbg!("Csum: {:.9}", poly);

    let exp = f64::exp(-beta);
    let exp_d_beta = -exp;
    dbg!("expbeta: {:.9}", exp);

    let value = exp * poly;
    let d_beta = exp_d_beta * poly + exp * poly_d_beta;
    dbg!("frho: {:.9}", value);

    // Reformulate back as a function of rhosq
    let d_rhosq = d_beta * beta_d_rhosq;
    (value, d_rhosq)
}

/// The repulsive part of the potential f(rho), but calculates rho from input distance vector
/// r, and unit normal n
fn crespi_f(params: &Params, distsq: f64, r: V3, n: V3) -> (f64, V3, V3) {
    // first we need the transverse distances (aka rho^2)
    let (rhosq, rhosq_d_r, rhosq_d_n) = rho_squared(distsq, r, n);

    // then we calculate f itself
    let (f, f_d_rhosq) = crespi_fp(params, rhosq);

    let value = f;
    let d_r = f_d_rhosq * rhosq_d_r;
    let d_n = f_d_rhosq * rhosq_d_n;

    (value, d_r, d_n)
}

/// The repulsive part of the potential.
fn repulsive_part(params: &Params, distsq: f64, r: V3, ni: V3, nj: V3) -> (f64, V3, V3, V3) {
    debug_assert_eq!(distsq, r.sqnorm());
//    debug_assert_close!(1.0, ni.sqnorm());  // untrue during numerical differentiation
//    debug_assert_close!(1.0, nj.sqnorm());  // untrue during numerical differentiation

    // calculate f(rho_ij) for ij and ji
    let (fij, fij_d_r, fij_d_ni) = crespi_f(params, distsq, r, ni);
    let (fji, fji_d_r, fji_d_nj) = crespi_f(params, distsq, r, nj);

    let value = params.C + fij + fji;
    let d_r = fij_d_r + fji_d_r;
    let d_ni = fij_d_ni;
    let d_nj = fji_d_nj;
    dbg!("VR: {:.9}", value);

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
                num_grad_v3(1e-3, rij, |rij| super::compute(params, rij, ni, nj).value).0,
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
                num_grad_v3(1e-5, ni, |ni| super::compute(params, rij, ni, nj).value).0,
            }

            let tol = get_n_tol(nj);
            assert_close!{
                rel=tol, abs=tol, grad_nj.0,
                num_grad_v3(1e-5, nj, |nj| super::compute(params, rij, ni, nj).value).0,
            }
        }
    }

    #[test]
    fn attractive_part() {
        // super::attractive_part(dist_sq: f64, r: V3)
        let ref params = Params::default();

        for _ in 0..NTRIAL {
            let rij = params.random_r();
            let (_, d_rij) = super::attractive_part(params, rij.sqnorm(), rij);

            assert_close!{
                rel=1e-11, abs=1e-11, d_rij.0,
                num_grad_v3(1e-3, rij, |rij| super::attractive_part(params, rij.sqnorm(), rij).0).0,
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
            let (_, d_rhosq) = super::crespi_fp(params, rhosq);

            assert_close!{
                rel=1e-10, abs=1e-10, d_rhosq,
                numerical::slope(1e-3, None, rhosq, |rhosq| super::crespi_fp(params, rhosq).0),
            }
        }
    }

    #[test]
    fn crespi_f() {
        let ref params = Params::default();
        for _ in 0..NTRIAL {
            let r = params.random_r();
            let n = V3::random_unit();
            let (_, d_r, d_n) = super::crespi_f(params, r.sqnorm(), r, n);

            assert_close!{
                rel=1e-11, abs=1e-11, d_r.0,
                num_grad_v3(1e-3, r, |r| super::crespi_f(params, r.sqnorm(), r, n).0).0,
            }

            // Finite differences w.r.t. n seem unreliable when n is in the vicinity of r_hat
            let n_tol = match V3::dot(&r.unit(), &n).abs() {
                x if x > 0.97 => 1e-7,
                _ => 1e-11,
            };
            assert_close!{
                rel=n_tol, abs=n_tol, d_n.0,
                num_grad_v3(1e-5, n, |n| super::crespi_f(params, r.sqnorm(), r, n).0).0,
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

            let (_, d_rij, d_ni, d_nj) = super::repulsive_part(params, rij.sqnorm(), rij, ni, nj);

            assert_close!{
                rel=1e-11, abs=1e-11, d_rij.0,
                num_grad_v3(1e-3, rij, |rij| super::repulsive_part(params, rij.sqnorm(), rij, ni, nj).0).0,
            }

            // Finite differences w.r.t. n seem unreliable when n is in the vicinity of r_hat
            let get_n_tol = |n| match V3::dot(&rij.unit(), &n).abs() {
                x if x > 0.97 => 1e-7,
                _ => 1e-11,
            };

            let tol = get_n_tol(ni);
            assert_close!{
                rel=tol, abs=tol, d_ni.0,
                num_grad_v3(1e-5, ni, |ni| super::repulsive_part(params, rij.sqnorm(), rij, ni, nj).0).0,
                "unitdot: {}", V3::dot(&rij.unit(), &ni).abs(),
            }

            let tol = get_n_tol(nj);
            assert_close!{
                rel=tol, abs=tol, d_nj.0,
                num_grad_v3(1e-5, nj, |nj| super::repulsive_part(params, rij.sqnorm(), rij, ni, nj).0).0,
                "unitdot: {}", V3::dot(&rij.unit(), &nj).abs(),
            }
        }
    }
}

//------------------------------------------------------------------

#[cfg(test)]
#[derive(Deserialize)]
struct ForceFile {
    value: f64,
    grad: Vec<V3>,
}

#[cfg(test)]
#[deny(unused)]
mod input_tests {
    use super::*;
    use crate::FailResult;
    use std::{path::Path, fs::File, io};
    use rsp2_structure_io::Poscar;
    use rsp2_structure::{
        bonds::FracBonds,
        consts::CARBON,
    };
    use rsp2_array_types::Unvee;

    const RESOURCE_DIR: &'static str = "tests/resources";
    const BIG_INPUT_1: &'static str = "tblg-2011-150-a";

    #[test]
    fn all() -> FailResult<()> {
        let mut matches = vec![];
        for entry in Path::new(RESOURCE_DIR).join("crespi").read_dir()? {
            let entry: String = entry?.path().display().to_string();
            if let Some(base) = strip_suffix(".lmp.json.xz", &entry) {
                matches.push(Path::new(&base).file_name().unwrap().to_string_lossy().into_owned());
            }
        }
        assert!(!matches.is_empty(), "failed to locate test inputs!");

        for name in matches {
            if name != BIG_INPUT_1 {
                println!("Testing {}", name);
                single(&name)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn big_input_1() -> FailResult<()> {
        single(BIG_INPUT_1)
    }

    fn single(name: &str) -> FailResult<()> {
        let ref mut params = Params::default();
        params.cutoff_transition_dist = None; // Lammps has no smooth cutoff
        params.cutoff_begin = 14.0;

        let in_path = Path::new(RESOURCE_DIR).join("structure").join(name).join("structure.vasp.xz");
        let layers_path = Path::new(RESOURCE_DIR).join("structure").join(name).join("layers.json.xz");
        let out_path = Path::new(RESOURCE_DIR).join("crespi").join(name.to_string() + ".lmp.json.xz");

        let expected: ForceFile = serde_json::from_reader(open_xz(out_path)?)?;
        let ref layers: Vec<i32> = serde_json::from_reader(open_xz(layers_path)?)?;
        let Poscar { ref mut coords, ref elements, .. } = Poscar::from_reader(open_xz(in_path)?)?;

        let bonds = FracBonds::compute_with_meta(
            coords,
            zip_eq!(elements, layers),
            |(&ea, &la), (&eb, &lb)| match ((ea, eb), i32::abs(la - lb)) {
                ((CARBON, CARBON), 1) => Some(14.0 * 1.001),
                _ => None,
            },
        )?;

        let mut value = 0.0;
        let mut grad = vec![V3::zero(); coords.len()];

        let cart_coords = coords.with_carts(coords.to_carts());
        for frac_bond in &bonds {
            if !frac_bond.is_canonical() {
                continue;
            }
            let rij = frac_bond.cart_vector_using_cache(&cart_coords).unwrap();
            let (bond_value, bond_d_rij) = params.compute_z(rij);
            value += bond_value;
            grad[frac_bond.from] -= bond_d_rij;
            grad[frac_bond.to] += bond_d_rij;
        }

        assert_close!(abs=1e-7, rel=1e-6, value, expected.value, "in file: {}", name);
        assert_close!(abs=1e-7, rel=1e-6, grad.unvee(), expected.grad.unvee(), "in file: {}", name);
        Ok(())
    }

    fn open_xz(path: impl AsRef<Path>) -> io::Result<impl io::Read> {
        File::open(path).map(::xz2::read::XzDecoder::new)
    }

    fn strip_suffix<'a>(suffix: &str, s: &str) -> Option<String> {
        if s.ends_with(suffix) {
            Some(s[..s.len()-suffix.len()].to_string())
        } else {
            None
        }
    }
}
