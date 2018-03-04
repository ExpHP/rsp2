#![allow(deprecated)] // HACK: reduce warning-spam at call sites, I only care about the `use`

use ::{Lattice, Coords};
#[warn(deprecated)]
use ::rsp2_array_utils::{dot, det, map_arr, map_mat};
use super::reduction::LatticeReduction;

use ::rsp2_array_types::{V3, Unvee};

pub fn lattice_point_group(
    reduction: &LatticeReduction,
    tol: f64,
) -> Vec<[[i32; 3]; 3]>
{
    Context {
        lattice: reduction.clone(),
        tol
    }.lattice_point_group()
}

// TODO: need to chase down Le Page, Y. (1982).J. Appl. Cryst.15, 255-259.
//       to find its proof of why only linear combinations up to absolute
//       value 2 need to be considered for twofold rotations.

//       (especially considering that we plan to search for more than
//        just twofolds!)

//       My current assumption is that, for reduced lattices, the points whose
//       coordinates lie within absolute value 2 are the only possible
//       points that can possibly be equal in length to a lattice vector.


struct Context {
    lattice: LatticeReduction,
    tol : f64,
}

impl Context {

    fn lattice_point_group(&self) -> Vec<[[i32; 3]; 3]>
    {
        // coefficient matrix;  L = C L_reduced
        let c_mat = self.lattice.transform().inverse_matrix();
        let c_inv = self.lattice.transform().matrix();

        self.reduced_lattice_point_group()
            .iter()
            .map(|m| dot(c_mat, &dot(m, c_inv)))
            .collect()
    }

    fn reduced_lattice_point_group(&self) -> Vec<[[i32; 3]; 3]>
    {
        // For a given lattice, each rotation operator has a corresponding
        //  unimodular transform:
        //
        //   (forall R. exists σ unimodular.)  L R^T = σ L
        //
        // It is easy to show that σ must satisfy:  (σ L) (σ L)^T == L L^T
        //
        // From the diagonal elements of this equality, we see that the 'kth' row
        // of (σ L) must be equal in length to the kth row of L.
        // (but of course; a rotation must not change a vector's length!)
        //
        // This gives us an *extremely* small search space for valid rotations.
        let lengths = self.lattice.reduced().norms();
        let choices_frac = map_arr(lengths, |x| self.lattice_points_of_length(x));
        let choices_cart = map_arr(choices_frac.clone(), |ref choices| {
                Coords::Fracs(floatify(choices))
                    .to_carts(self.lattice.reduced())
        });

        // off diagonal elements of L L^T
        let metric_off_diags = |m: &[V3; 3]| [
            V3::dot(&m[1], &m[2]),
            V3::dot(&m[2], &m[0]),
            V3::dot(&m[0], &m[1]),
        ];
        let target_off_diags = metric_off_diags(self.lattice.reduced().vectors());

        // Build unimodular matrices from those choices
        let mut unimodulars = vec![];
        for (&frac_0, &cart_0) in izip!(&choices_frac[0], &choices_cart[0]) {
            for (&frac_1, &cart_1) in izip!(&choices_frac[1], &choices_cart[1]) {
                // we *could* filter on the cross product of
                // these rows (if it's 0, so is the determinant), but meh.
                for (&frac_2, &cart_2) in izip!(&choices_frac[2], &choices_cart[2]) {

                    // Most of these matrices won't be unimodular; filter them out.
                    let unimodular = [frac_0, frac_1, frac_2].unvee();
                    if det(&unimodular).abs() != 1 {
                        continue;
                    }

                    // Check the off-diagonal elements of the metric.
                    // (this completes verification that (σ L) (σ L)^T == L L^T)
                    let off_diags = metric_off_diags(&[cart_0, cart_1, cart_2]);

                    // NOTE: might need to revisit how tolerance is applied here.
                    //       Absolute and relative tolerance both look bad;
                    //       the quantities we are looking at could very well
                    //        come out to ~zero after nontrivial cancellations.

                    let eff_tol = 1e-5 * self.lattice.reduced().volume().cbrt();

                    if (0..3).all(|k| (off_diags[k] - target_off_diags[k]).abs() <= eff_tol) {
                        unimodulars.push(unimodular);
                    }
                }
            }
        }

        let l_inv = self.lattice.reduced().inverse_matrix();
        let l_mat = self.lattice.reduced().matrix();

        unimodulars.into_iter()
            .map(move |u| map_mat(u, |x| x as f64))
            .map(|u| dot(l_inv, &dot(&u, l_mat)))
            // FIXME: when might this fail? (bug? user error?)
            .map(|u| ::util::Tol(1e-3).unfloat_33(&u).unwrap())
            .collect()
    }

    fn lattice_points_of_length(&self, target_length: f64) -> Vec<V3<i32>>
    {
        Coords::Fracs(LATTICE_POINTS_FLOAT.clone()).to_carts(&self.lattice.reduced())
            .into_iter()
            .map(|v| v.norm())
            .enumerate()
            .filter(|&(_, r)| (r - target_length).abs() < self.tol * target_length)
            .map(|(i, _)| LATTICE_POINTS_INT[i])
            .collect()
    }
}

lazy_static!{
    // a set of fractional lattice coordinates large enough that,
    // for a reduced cell, this will include all vectors equal in length
    // to a cell vector
    static ref LATTICE_POINTS_INT: Vec<V3<i32>> = {
        // FIXME: this is a fairly large region for the sake of paranoia
        //         until I can find and verify Le Page's proof.
        const MAX: i32 = 5;
        let mut indices = Vec::with_capacity((2 * MAX + 1).pow(3) as usize);
        for i in -MAX..MAX + 1 {
            for j in -MAX..MAX + 1 {
                for k in -MAX..MAX + 1 {
                    indices.push(V3([i, j, k]));
                }
            }
        }
        indices
    };

    static ref LATTICE_POINTS_FLOAT: Vec<V3> = floatify(&LATTICE_POINTS_INT);
}

fn floatify(vs: &[V3<i32>]) -> Vec<V3>
{ vs.iter().map(|&v| v.map(|x| x.into())).collect() }
