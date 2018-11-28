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

use rsp2_structure::{CoordsKind, Lattice, Coords};
use rsp2_kets::{Ket, KetRef, Rect};
use rsp2_array_utils::{arr_from_fn};
use rsp2_array_types::{V3, M33, dot, inv};
use crate::threading::Threading;

use rayon::prelude::*;
use std::f64::consts::PI;
use itertools::Itertools;

//---------------------------
// NOTE: (2018-02-16) (gamma-point)
//
// Historically this code was intended to be used for unfolding
// bands at a user-specified Q point in the primitive reciprocal FBZ.
// However, this implementation appears to be buggy for non-gamma
// q-points, and I was never able to find the issue.
//
// While I *could* try to pull out all the details related to
// user-specified Q-points, it would affect a large portion of the code
// and I'd be tempted to rewrite half of the thing.
//
// Hence, I have chosen to leave these details in---correct or not---and
// simply limit the API to only support gamma point.
//  - ML
//---------------------------

pub use self::config::Config;
pub mod config {
    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[serde(rename_all = "kebab-case")]
    pub struct Config {
        pub fbz: FbzType,
        pub sampling: SampleType,
    }

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[derive(Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    #[serde(rename_all = "kebab-case")]
    pub enum FbzType {
        /// Parallelepiped defined by the reciprocal vectors.
        ReciprocalCell,

        /// Voronoi cell in reciprocal space.  I.e. the ACTUAL Brillouin Zone.
        ///
        /// I don't think you need this. Also, it's not implemented.
        /// This is only here to document the fact that this code does not
        /// use the true FBZ.
        Voronoi,
    }

    /// How the primitive cell reciprocal lattice is sampled.
    ///
    /// Basically, contributions are added together from various images of the
    /// eigenvector q point under the primitive cell reciprocal lattice before
    /// computing the final probabilities.
    ///
    /// Even for perfect supercells, having too few sampling points may lead to
    /// probabilities that are too small to provide any information. (e.g. consider
    /// a breathing mode of a primitive cell at gamma, where the total sum of all
    /// atomic displacement vectors in the mode is the zero vector. Sampling this
    /// mode at the true primitive gamma would likewise produce a dot product of zero)
    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[derive(PartialEq, Eq, Hash)]
    #[serde(rename_all = "kebab-case")]
    pub enum SampleType {
        /// Sample kpoints with primitive reciprocal coords in the cartesian product
        /// `(-nx...nx) × (-ny...ny) × (-nz...nz)`
        ///
        /// This is what Zheng/Zhang (2017) do but I am not certain if
        /// it provides any conceivable benefit.
        Centered([u32; 3]),
        /// Sample kpoints with primitive reciprocal coords in the cartesian product
        /// `(0..nx) × (0..ny) × (0..nz)`
        Plain([u32; 3]),
    }
}

impl self::config::SampleType {
    fn signed_indices(&self) -> Vec<V3<i32>>
    {
        use self::config::SampleType::*;
        let ax = |k: usize| match *self {
            Centered(arr) => -(arr[k] as i32)..arr[k] as i32+1,
            Plain(arr) => 0..arr[k] as i32,
        };

        iproduct!(ax(0), ax(1), ax(2)).map(|(i,j,k)| V3([i,j,k])).collect()
    }

    fn points(&self, lattice: &M33) -> Vec<V3>
    {
        self.signed_indices().into_iter()
            .map(|a| a.map(f64::from))
            .map(|v| v * lattice)
            .collect::<Vec<_>>()
    }
}

/// A supercell matrix which is not necessarily diagonal,
/// but for which it is already known how many images are
/// needed along each axis in order to uniquely describe
/// all images of the primitive cell.
///
/// This only exists for now because I am still too lazy to
/// properly incorporate an HNF search into this codebase...
#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct ScMatrix {
    pub matrix: M33<i32>,
    pub periods: [u32; 3],
}

impl ScMatrix {
    pub fn new(matrix: &M33<i32>, periods: &[u32; 3]) -> Self
    {
        // sanity check
        // (NOTE: this condition is necessary, but not sufficient)
        // (NOTE: this is obviously not considered to be an invariant of this type,
        //        due to public members and Deserialize)
        assert_eq!(matrix.det().abs() as u32, periods.iter().product::<u32>());
        ScMatrix { matrix: *matrix, periods: *periods }
    }
}

/// # Output
///
/// Probabilities associated with each image of `eigenvector_q` under the supercell
/// reciprocal lattice that is distinct under the primitive reciprocal lattice.
///
/// The `u32` array contains the integer coordinates of the supercell reciprocal
/// lattice vector that is to be added to `eigenvector_q` to produce the image.
///
/// # Citations
///
/// This code is largely based on Zheng, Fawei; Zhang, Ping (2017),
/// “Phonon Unfolding: A program for unfolding phonon dispersions of materials”,
///   Mendeley Data, v1 http://dx.doi.org/10.17632/3hpx6zmxhg.1
// NOTE: This only exists for convenience.
//       If you have many eigenvectors, use GammaUnfolder directly for
//       much better performance.
#[allow(unused)]
pub fn unfold_gamma_phonon(
    config: &Config,
    threading: Threading,
    // Takes CoordStructure because I think there might be a case for
    // supporting <M: Eq + Hash>, with the semantics that atoms with
    // non-equal metadata are "distinct" and contributions between
    // them to a projection cannot cancel.
    superstructure: &Coords,
    // eigenvector_q: &SuperFracQ, // NOTE: only gamma now
    eigenvector: KetRef<'_>,
    supercell_matrix: &ScMatrix,
) -> Vec<([u32; 3], f64)>
{
    let unfolder = GammaUnfolder::from_config(config, threading, superstructure, supercell_matrix);
    let indices = unfolder.q_indices().iter().cloned();
    let probs = unfolder.unfold_phonon(threading, eigenvector);
    izip!(indices, probs).collect()
}

/// Contains precomputed information derived from the
/// q-points at which integration will be performed;
///
/// This speeds up the unfolding of many eigenvectors computed
/// at the same q point.
pub struct GammaUnfolder {
    /// `[sc_recip_index] -> index` (3D index of sc reciprocal lattice point)
    sc_indices: Vec<[u32; 3]>,
    /// `[pc_recip_index][sc_recip_index] -> q_ket`
    q_kets_by_pc_sc: Vec<Vec<Ket>>,

    // HACK: these are only here to service the code that uses UnfolderAtQ
    /// `[sc_recip_index] -> q` (reduced into primitive cell)
    sc_qs_frac: Vec<PrimFracQ>,
}

impl GammaUnfolder {
    pub fn from_config(
        config: &Config,
        threading: Threading,
        superstructure: &Coords,
        sc_matrix: &ScMatrix,
        // eigenvector_q: &V3, // reduced by sc lattice
    ) -> GammaUnfolder
    {
        // HACK: ctrl-F "gamma-point" for more info
        let eigenvector_q = &V3([0.0; 3]);

        // FIXME expected behavior unclear when the following does not hold.
        //       especially so if it lies outside the (larger) primitive reciprocal cell.
        assert!(eigenvector_q.iter().all(|&x| 0.0 <= x && x < 1.0));

        match config.fbz {
            config::FbzType::Voronoi => {
                panic!("fbz type Voronoi not supported (yet) (or possibly ever)")
            },

            config::FbzType::ReciprocalCell => {
                // Generate a bunch of q points indexed by:
                //  - primitive reciprocal lattice vectors (which all contribute)
                //  - supercell reciprocal lattice vectors (which we are trying to project onto)
                let sc_lattice = superstructure.lattice().matrix();
                let sc_inverse = superstructure.lattice().inverse_matrix();
                let ref pc_lattice = &inv(&sc_matrix.matrix.map(|x| x as f64)) * sc_lattice;
                let ref pc_inverse = inv(pc_lattice);
                let ref sc_recip = sc_inverse.t();
                let ref pc_recip = pc_inverse.t();

                // lattice points of interest
                let sc_periods = sc_matrix.periods;

                let quotient_sample_spec = self::config::SampleType::Plain(sc_periods);
                let quotient_indices: Vec<_> =
                    quotient_sample_spec.signed_indices()
                        .into_iter()
                        .map(|v| {
                            arr_from_fn(|k| (v[k] + sc_periods[k] as i32) as u32 % sc_periods[k])
                        }).collect();
                assert!(quotient_indices.len() > 0, "no points to sample against");

                let quotient_vecs = quotient_sample_spec.points(sc_recip);
                let pc_recip_vecs = config.sampling.points(pc_recip);

                // into recip cartesian space
                let eigenvector_q_cart = eigenvector_q * sc_recip;
                if eigenvector_q != &V3([0.0; 3]) {
                    // (I currently always run this code on gamma eigenvectors...)
                    warn!("Untested code path: 9fc15058-7199-45d2-80ec-630ceb575d3d");
                }

                GammaUnfolder {
                    sc_indices: quotient_indices,
                    q_kets_by_pc_sc: {
                        threading.maybe_serial(|| {
                                pc_recip_vecs.par_iter().map(|sample_q| {
                                    quotient_vecs.par_iter().map(|quotient_q| {
                                        q_ket(
                                            &superstructure.to_carts(),
                                            &(eigenvector_q_cart + sample_q + quotient_q),
                                        )
                                    }).collect()
                                }).collect()
                            },
                        )
                    },
                    sc_qs_frac: {
                        let pc_recip = Lattice::new(pc_recip);
                        CoordsKind::Carts(quotient_vecs.clone()).to_fracs(&pc_recip)
                    },
                }
            },
        }
    }

    #[allow(unused)]
    pub fn q_indices(&self) -> &[[u32; 3]]
    { &self.sc_indices }

    #[allow(unused)]
    pub fn q_fracs(&self) -> &[V3]
    { &self.sc_qs_frac }

    pub fn unfold_phonon(&self, threading: Threading, eigenvector: KetRef<'_>) -> Vec<f64>
    {
        assert_eq!(eigenvector.len(), 3 * self.q_kets_by_pc_sc[0][0].len());

        // separate kets for each polarization axis (they don't cancel each other)
        let axis_evs: [Ket; 3] = arr_from_fn(|axis| {
            eigenvector.iter().tuples()
                .map(|(x, y, z)| [x, y, z][axis])
                .collect()
        });

        let num_sc_recip_vecs = self.sc_indices.len();

        // different SC reciprocal lattice vectors compete with each other.
        // different PC reciprocal lattice vectors work together.
        let mut probs = vec![0.0f64; num_sc_recip_vecs];
        for q_kets_by_sc in &self.q_kets_by_pc_sc {
            threading.maybe_serial(|| {
                assert_eq!(probs.len(), q_kets_by_sc.len());
                probs.par_iter_mut()
                    .zip_eq(q_kets_by_sc)
                    .for_each(|(prob, q_ket)| {
                        for k in 0..3 {
                            *prob += q_ket.as_ref().overlap(&axis_evs[k]);
                        }
                    })
            });
        }

        // unfortunately it seems this method of computing weights is rather
        // fuzzy and the total weight will tend to be something completely
        // unpredictable; hence we must normalize it after the fact.
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);

        probs
    }
}

// NOTE: a Ket of length N_a rather than 3 * N_a
fn q_ket(carts: &[V3], q: &V3) -> Ket
{ carts.iter().map(|x| Rect::from_phase(-dot(x, q) * 2.0 * PI)).collect() }

#[allow(unused)]
type SuperFracQ = V3;
type PrimFracQ = V3;

#[cfg(test)]
#[deny(dead_code)]
mod tests {
    use super::*;
    use slice_of_array::prelude::*;
    use rsp2_structure::{CoordsKind, Lattice};
    use rsp2_array_types::{Envee, mat};

    #[test]
    fn simple_unfold() {
        fn do_it(
            structure: &Coords,
            sc_vec: V3<i32>,
            expect_index: &[[u32; 3]],
            eigenvector: Vec<V3>,
        ) {

            let configs = vec![
                from_json!({
                    "fbz": "reciprocal-cell",
                    "sampling": { "plain": [3, 3, 3] },
                }),
                from_json!({
                    "fbz": "reciprocal-cell",
                    "sampling": { "centered": [1, 1, 1] },
                }),
            ];
            let eigenvector: Ket = eigenvector.flat().iter().map(|&r| Rect::from(r)).collect();
            let sc_mat = ScMatrix::new(
                &mat::from_array([[sc_vec[0], 0, 0], [0, sc_vec[1], 0], [0, 0, sc_vec[2]]]),
                &sc_vec.map(|x| x as u32),
            );
            for config in &configs {
                let unfolded = unfold_gamma_phonon(
                    config,
                    Threading::Serial,
                    structure,
                    eigenvector.as_ref(),
                    &sc_mat,
                );

                // expect that all of the nonzero probability is distributed among
                // those entries in expect_index, and that the total is 1
                let (mut ayes, mut nays) = (0.0, 0.0);
                for &(index, p) in &unfolded {
                    match expect_index.contains(&index) {
                        true => ayes += p,
                        false => nays += p,
                    }
                }
                assert!(
                    (ayes - 1.0).abs() < 1e-6 && nays < 1e-6,
                    "{:?} {:?}", expect_index, unfolded,
                );
            }
        };

        //--------------------------------------------
        // easy 1D case
        // 1 atom per primitive cell
        let structure = Coords::new(
            Lattice::diagonal(&[1.0, 1.0, 4.0]),
            CoordsKind::Carts(vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ].envee()),
        );
        let sc_vec = V3([1, 1, 4]);
        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        go_do_it(&[[0, 0, 0]], vec![
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ].envee());

        // phase rotation of 2/4 tau (i.e. `-1`) per unit cell
        go_do_it(&[[0, 0, 2]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
        ].envee());

        // phase rotation of 1/4 tau (i.e. `i`) per unit cell.
        // because gamma eigenvectors are always real, we end up
        //  with contributions from two kpoints whose imaginary
        //  parts cancel.
        go_do_it(&[[0, 0, 1], [0, 0, 3]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ].envee());

        //--------------------------------------------
        // supercell along multiple dimensions
        // 1 atom per primitive cell
        let structure = Coords::new(
            Lattice::diagonal(&[2.0, 2.0, 1.0]),
            CoordsKind::Carts(vec![
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ].envee()),
        );
        let sc_vec = V3([2, 2, 1]);

        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        // gamma
        let p = 8_f64.sqrt().recip();
        go_do_it(&[[0, 0, 0]], vec![
            [ p,  p, 0.0],
            [ p,  p, 0.0],
            [ p,  p, 0.0],
            [ p,  p, 0.0],
        ].envee());

        // non-gamma along one axis
        let p = 8_f64.sqrt().recip();
        go_do_it(&[[0, 1, 0]], vec![
            [-p,  p, 0.0],
            [ p, -p, 0.0],
            [-p,  p, 0.0],
            [ p, -p, 0.0],
        ].envee());

        // non-gamma along multiple axis.
        let p = 8_f64.sqrt().recip();
        go_do_it(&[[1, 1, 0]], vec![
            [-p,  p, 0.0],
            [ p, -p, 0.0],
            [ p, -p, 0.0],
            [-p,  p, 0.0],
        ].envee());

        //--------------------------------------------
        // non-diagonal lattice.
        // hopefully, this will catch bugs involving incorrect
        //   usage of matrices vs their transpose.
        // 1 atom per primitive cell
        let structure = Coords::new(
            Lattice::from(&[
                [1.0, 0.0, 0.0],
                [-0.5, 0.5 * 3_f64.sqrt(), 0.0],
                [0.0, 0.0, 1.0],
            ]),
            CoordsKind::Fracs(vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ].envee()),
        );
        let sc_vec = V3([2, 2, 1]);

        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        go_do_it(&[[1, 1, 0]], vec![
            [0.0, 0.0, -0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
        ].envee());

        //--------------------------------------------
        // primitive structure with more than one atom
        let structure = Coords::new(
            Lattice::from(&[
                // graphene cell (2 atoms per primitive),
                // doubled along b (4 atoms per supercell)
                [1.0, 0.0, 0.0],
                [-1.0, 1.0 * 3_f64.sqrt(), 0.0],
                [0.0, 0.0, 1.0],
            ]),
            CoordsKind::Fracs(vec![
                // honeycomb pattern
                [    0.0, 0.0, 0.0],
                [1.0/3.0, 0.0, 0.0],
                [    0.0, 0.5, 0.0],
                [1.0/3.0, 0.5, 0.0],
            ].envee()),
        );
        let sc_vec = V3([1, 2, 1]);

        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        // the obvious gamma vec
        go_do_it(&[[0, 0, 0]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
        ].envee());

        // the less obvious gamma vec
        // (sign only changes within the primitive cell)
        //
        // If we did not sample other primitive reciprocal cell vectors
        // beyond the true gamma, this would fail.
        go_do_it(&[[0, 0, 0]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
        ].envee());

        // a non-gamma vec
        go_do_it(&[[0, 1, 0]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ].envee());

        //--------------------------------------------

        // TODO: Test with non-perfect supercell
        // TODO: Test with non-diagonal supercell
    }
}
