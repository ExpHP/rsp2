use rsp2_array_types::{V3, M33, M3};
use rsp2_soa_ops::{Perm, Permute};

use std::os::raw::c_char;
use std::ffi::CStr;

use slice_of_array::prelude::*;
use rayon::prelude::*;
use num_complex::Complex64;

#[macro_use] extern crate rsp2_util_macros;
#[macro_use] extern crate rsp2_assert_close;

#[no_mangle]
pub extern "C" fn rsp2c_unfold_all(
    num_quotient: i64,
    num_sites: i64,
    num_evecs: i64,
    progress_prefix: *const c_char, // NUL-terminated UTF-8, possibly NULL
    super_lattice: *const f64, // shape (3, 3)
    super_carts: *const f64, // shape (sites, 3)
    translation_carts: *const f64, // shape (quotient, 3)
    gpoint_sfracs: *const f64, // shape (quotient, 3)
    kpoint_sfrac: *const f64, // shape (3,)
    eigenvectors: *const Complex64, // shape (evecs, sites, 3)
    translation_deperms: *const i32, // shape (quotient, sites)
    output: *mut f64, // shape (evecs, quotient)
    // return is nonzero on error
) -> i32 {
    match std::panic::catch_unwind(|| {
        let num_quotient = num_quotient as usize;
        let num_sites = num_sites as usize;
        let num_evecs = num_evecs as usize;

        use std::slice::{from_raw_parts, from_raw_parts_mut};
        unsafe {
            let ref super_lattice = M3(*from_raw_parts(super_lattice, 3 * 3).nest().as_array());
            let super_carts = from_raw_parts(super_carts, num_sites * 3).nest();
            let translation_carts = from_raw_parts(translation_carts, num_quotient * 3).nest();
            let gpoint_sfracs = from_raw_parts(gpoint_sfracs, num_quotient * 3).nest();
            let kpoint_sfrac = from_raw_parts(kpoint_sfrac, 3).as_array();
            let eigenvectors = from_raw_parts(eigenvectors, num_evecs * num_sites * 3).nest();
            let translation_deperms = from_raw_parts(translation_deperms, num_quotient * num_sites);
            let output = from_raw_parts_mut(output, num_evecs * num_quotient);

            let progress_prefix = if progress_prefix.is_null() {
                None
            } else {
                Some(CStr::from_ptr(progress_prefix).to_str().unwrap())
            };

            unfold_all(
                progress_prefix,
                super_lattice,
                super_carts,
                translation_carts,
                gpoint_sfracs,
                kpoint_sfrac,
                eigenvectors,
                translation_deperms,
                output,
            );
        }
    }) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

fn unfold_all(
    progress_prefix: Option<&str>,
    super_lattice: &M33,
    super_carts: &[V3],
    translation_carts: &[V3],
    gpoint_sfracs: &[V3],
    kpoint_sfrac: &V3,
    eigenvectors: &[V3<Complex64>],
    translation_deperms: &[i32],
    output: &mut [f64],
) {
    let num_sites = super_carts.len();
    let num_quotient = translation_carts.len();

    let super_lattice_inv = M33::inv(super_lattice);
    let ref translation_sfracs: Vec<_> = translation_carts.iter().map(|v| v * super_lattice_inv).collect();

    let ref translation_deperms: Vec<Perm> = {
        translation_deperms.chunks(num_sites)
            .map(|perm| Perm::from_vec(perm.iter().map(|&i| i as usize).collect()).expect("invalid perm!"))
            .collect()
    };

    let translation_phases: Vec<Vec<Complex64>> = {
        let super_lattice_recip = super_lattice_inv.t();
        let kpoint_cart = kpoint_sfrac * super_lattice_recip;
        zip_eq!(translation_carts, translation_deperms)
            .map(|(&translation_cart, translation_deperm)| get_translation_phases(
                kpoint_cart,
                super_carts,
                translation_cart,
                translation_deperm,
            ))
            .collect()
    };

    let progress = progress_prefix.map(|prefix| {
        move |done, total| println!("{}Unfolded {:>5} of {} eigenvectors", prefix, done, total)
    });

    let iter = zip_eq!(eigenvectors.chunks(num_sites), output.chunks_mut(num_quotient)).enumerate();
    let num_total = iter.len();
    for (num_complete, (eigenvector, output)) in iter {
        if let Some(progress) = progress.as_ref() {
            progress(num_complete, num_total);
        }

        let dense_row = unfold_one(
            translation_sfracs,
            translation_deperms,
            &translation_phases,
            gpoint_sfracs,
            kpoint_sfrac,
            eigenvector,
        );
        output.copy_from_slice(&dense_row);
    }

    if let Some(progress) = progress {
        progress(num_total, num_total);
    }
}

fn unfold_one(
    translation_sfracs: &[V3],
    translation_deperms: &[Perm],
    translation_phases: &[Vec<Complex64>],
    gpoint_sfracs: &[V3],
    kpoint_sfrac: &V3,
    eigenvector: &[V3<Complex64>],
) -> Vec<f64> {
    let num_quotient = translation_sfracs.len();

    let inner_prods: Vec<_> = {
        translation_deperms.par_iter().zip_eq(translation_phases).map(|(perm, image_phases)| {
            let permuted = eigenvector.to_vec().permuted_by(perm);
            zip_eq!(eigenvector, image_phases, permuted)
                .map(|(orig_v3, &image_phase, perm_v3)| {
                    let true_translated_v3 = perm_v3.map(|x| x * image_phase);
                    inner_prod_v3(orig_v3, &true_translated_v3)
                })
                .sum::<Complex64>()
        }).collect()
    };

    let gpoint_probs: Vec<_> = {
        gpoint_sfracs.par_iter().map(|g| {
            // SBZ kpoint dot r for every r
            let phases: Vec<_> = {
                translation_sfracs.iter()
                    .map(|t| exp_i2pi(-V3::dot(&(kpoint_sfrac + g), t)))
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

const TWO_PI_I: Complex64 = Complex64 { re: 0.0, im: 2.0 * std::f64::consts::PI };
fn exp_i2pi(x: f64) -> Complex64 {
    Complex64::exp(&(TWO_PI_I * x))
}

//======================================================================
// When we apply the translation operators, some atoms will map to images under the supercell that
// are different from the ones we have eigenvector data for. For kpoints away from supercell gamma,
// those images should have different phases in their eigenvector components.
//
// Picture that the supercell looks like this:
//
// Legend:
// - a diagram of integers (labeled "Indices") depicts coordinates, by displaying the number `i`
//   at the position of the `i`th atom.
// - a diagram with letters depicts a list of metadata (such as elements or eigenvector components)
//   by arranging them starting from index 0 in the lower left, and etc. as if they were applied
//   to the coords in their original order.
// - Parentheses surround the position of the original zeroth atom.
//
//                 6  7  8                    g  h  i
//    Indices:    3  4  5     Eigenvector:   d  e  f
//              (0) 1  2                   (a) b  c
//
// Consider the translation that moves the 0th atom to the location originally at index 3.
// Applying the deperm to the eigenvector (to "translate" it by this vector) yields:
//
//                       d  e  f
//      Eigenvector:    a  b  c
//                    (g) h  i
//
// In this example, g, h, and i do not have the correct phases because those atoms mapped to
// different images. To find the superlattice translation that describes these images, we must look
// at the coords.  First, translate the coords by literally applying the translation.  Then, apply
// the inverse coperm to make the indices match their original sites.
//
//   (applying translation...)      (...then applying inverse coperm)
//                  6  7  8                     0  1  2
//                 3  4  5                     6  7  8
//    Indices:    0  1  2         Indices:    3  4  5
//              (x) x  x                    (x) x  x
//
// If you subtract the original coordinates from these, you get a list of metadata describing the
// super-lattice translations for each site in the permuted structure; atoms 3..9 need no
// correction, while atoms 0..3 require a phase correction by some super-lattice vector R.
//
//                      R  R  R
//    Translations:    0  0  0
//                   (0) 0  0
//
fn get_translation_phases(
    kpoint_cart: V3,
    super_carts: &[V3],
    translation_cart: V3,
    translation_deperm: &Perm,
) -> Vec<Complex64> {
    let inverse_coperm = translation_deperm; // inverse of inverse

    let mut translated_carts = super_carts.to_vec().permuted_by(inverse_coperm);
    for v in &mut translated_carts {
        *v += translation_cart;
    }

    zip_eq!(translated_carts, super_carts)
        .map(|(new, &orig)| new - orig)
        .map(|r| exp_i2pi(V3::dot(&kpoint_cart, &r)))
        .collect()
}

fn inner_prod_ev(a: &[V3<Complex64>], b: &[V3<Complex64>]) -> Complex64 {
    zip_eq!(a, b).map(|(a, b)| inner_prod_v3(a, b)).sum()
}

fn inner_prod_v3(a: &V3<Complex64>, b: &V3<Complex64>) -> Complex64 {
    V3::from_fn(|i| a[i].conj() * b[i]).0.iter().sum()
}
