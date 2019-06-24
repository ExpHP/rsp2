use slice_of_array::prelude::*;
use rsp2_array_types::{V3, M33, M3};
use rsp2_soa_ops::{Perm, Permute};
use std::f64::consts::PI;
use num_complex::Complex64;
use rayon::prelude::*;

#[macro_use] extern crate rsp2_util_macros;
#[macro_use] extern crate rsp2_assert_close;

pub extern "C" fn rsp2c_unfold_all_gamma(
    num_quotient: i64,
    num_sites: i64,
    num_evecs: i64,
    super_lattice: *const f64, // shape (3, 3)
    super_carts: *const f64, // shape (sites, 3)
    translation_carts: *const f64, // shape (quotient, 3)
    gpoint_sfracs: *const f64, // shape (quotient, 3)
    eigenvectors: *const f64, // shape (evecs, sites, 3)
    translation_deperms: *const i32, // shape (quotient, sites)
    dest_csr: *mut Vec<f64>,
    // return is nonzero on error
) -> i32 {
    match std::panic::catch_unwind(|| {
        let num_quotient = num_quotient as usize;
        let num_sites = num_sites as usize;
        let num_evecs = num_evecs as usize;

        use std::slice::from_raw_parts;
        unsafe {
            let ref super_lattice = M3(*from_raw_parts(super_lattice, 3 * 3).nest().as_array());
            let super_carts = from_raw_parts(super_carts, num_sites * 3).nest();
            let translation_carts = from_raw_parts(translation_carts, num_sites * 3).nest();
            let gpoint_sfracs = from_raw_parts(gpoint_sfracs, num_quotient * 3).nest();
            let eigenvectors = from_raw_parts(eigenvectors, num_evecs * num_sites * 3).nest();
            let translation_deperms = from_raw_parts(translation_deperms, num_quotient * num_sites);
            let csr = unfold_all_gamma(
                num_evecs,
                super_lattice,
                super_carts,
                translation_carts,
                gpoint_sfracs,
                eigenvectors,
                translation_deperms,
            );

            std::mem::replace(dest_csr.as_mut().expect("unexpected null pointer"), csr);
        }
    }) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// :param superstructure: ``pymatgen.Structure`` object with `sites` sites.
/// :param supercell: ``Supercell`` object.
/// :param eigenvectors: Shape ``(num_evecs, 3 * sites)``, complex.
///
/// Each row is an eigenvector.  Their norms may be less than 1, if the
/// structure has been projected onto a single layer, but should not exceed 1.
/// (They will NOT be automatically normalized by this function, as projection
/// onto a layer may create eigenvectors of zero norm)
///
/// :param translation_deperms:  Shape ``(quotient, sites)``.
/// Permutations such that ``(carts + translation_carts[i])[deperms[i]]`` is
/// equivalent to ``carts`` under superlattice translational symmetry, where
/// ``carts`` is the supercell carts.
///
/// :param kpoint_sfrac: Shape ``(3,)``, real.
/// The K point in the SC reciprocal cell at which the eigenvector was computed,
/// in fractional coords.
///
/// :param progress: Progress callback.
/// Called as ``progress(num_done, num_total)``.
///
/// :return: Shape ``(num_evecs, quotient)``
/// For each vector in ``eigenvectors``, its projected probabilities
/// onto ``k + g`` for each g in ``supercell.gpoint_sfracs()``.
fn unfold_all_gamma(
    num_evecs: usize,
    super_lattice: &M33,
    super_carts: &[V3],
    translation_carts: &[V3],
    gpoint_sfracs: &[V3],
    eigenvectors: &[V3],
    translation_deperms: &[i32],
) -> Vec<f64> {
    let num_sites = super_carts.len();
    let super_lattice_inv = M33::inv(super_lattice);
    let ref translation_sfracs: Vec<_> = translation_carts.iter().map(|v| v * super_lattice_inv).collect();

    let ref translation_deperms: Vec<Perm> = {
        translation_deperms.chunks(num_sites)
            .map(|perm| Perm::from_vec(perm.iter().map(|&i| i as usize).collect()).expect("invalid perm!"))
            .collect()
    };

    let mut out = Vec::default();
    for i in 0..num_evecs {
        let eigenvector = &eigenvectors[i * 3 * num_sites..(i + 1) * 3 * num_sites];
        let ref eigenvector = eigenvector.iter().map(|v| v.map(|r| Complex64::new(r, 0.0))).collect::<Vec<_>>();

        let kpoint_sfrac = V3::zero();
        let dense_row = unfold_one(
            translation_sfracs,
            translation_deperms,
            gpoint_sfracs,
            kpoint_sfrac,
            eigenvector,
        );
        out.extend(dense_row);
    }
    out
}

fn unfold_one(
    translation_sfracs: &[V3],
    translation_deperms: &[Perm],
    gpoint_sfracs: &[V3],
    kpoint_sfrac: V3,
    eigenvector: &[V3<Complex64>],
) -> Vec<f64> {
    let num_quotient = translation_sfracs.len();

    let inner_prods: Vec<_> = {
        translation_deperms.iter().map(|perm| {
            let permuted = eigenvector.to_vec().permuted_by(perm);
            inner_prod_ev(eigenvector, &permuted)
        }).collect()
    };

    let gpoint_probs: Vec<_> = {
        gpoint_sfracs.par_iter().map(|g| {
            // SBZ kpoint dot r for every r
            let phases: Vec<_> = {
                let i = Complex64::i();
                translation_sfracs.iter()
                    .map(|t| Complex64::exp(&(-2.0 * PI * i * V3::dot(&(kpoint_sfrac + g), t))))
                    .collect()
            };
            let prob = zip_eq!(&inner_prods, phases).map(|(a, b)| a * b).sum::<Complex64>() / num_quotient as f64;

            // analytically, these are all real, positive numbers
            //
            // numerically, however, cancellation may cause issues
            assert!(f64::abs(prob.im) < 1e-7);
            assert!(-1e-7 < prob.re);

            f64::max(prob.re, 0.0)
        }).collect()
    };

    assert_close!(
        abs=1e-7,
        gpoint_probs.iter().sum::<f64>(),
        inner_prod_ev(eigenvector, eigenvector).norm(),
    );
    gpoint_probs
}

fn inner_prod_ev(a: &[V3<Complex64>], b: &[V3<Complex64>]) -> Complex64 {
    zip_eq!(a, b).map(|(a, b)| inner_prod_v3(a, b)).sum()
}

fn inner_prod_v3(a: &V3<Complex64>, b: &V3<Complex64>) -> Complex64 {
    V3::from_fn(|i| a[i].conj() * b[i]).0.iter().sum()
}

pub extern "C" fn rsp2c_vec_new() -> *mut Vec<f64> {
    Box::into_raw(Box::new(vec![]))
}

pub extern "C" fn rsp2c_vec_data(vec: *const Vec<f64>) -> *const f64 {
    unsafe { vec.as_ref() }.expect("unexpected null ptr").as_ptr()
}

pub extern "C" fn rsp2c_vec_len(vec: *const Vec<f64>) -> i64 {
    unsafe { vec.as_ref() }.expect("unexpected null ptr").len() as i64
}

pub extern "C" fn rsp2c_vec_free(vec: *mut Vec<f64>) {
    unsafe { drop(Box::from_raw(vec)) }
}
