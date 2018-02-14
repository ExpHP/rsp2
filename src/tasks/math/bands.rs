
use ::rsp2_structure::{Coords, Lattice, CoordStructure};
use ::rsp2_array_utils::{dot, map_arr};
use ::rsp2_kets::{Ket, KetRef, Rect};
use ::rsp2_array_utils::{arr_from_fn};


use ::std::f64::consts::PI;
use ::itertools::Itertools;

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

        /// Voronoi cell in reciprocal space.
        ///
        /// I don't think you need this.
        /// Also, it's not implemented.
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
    fn signed_indices(&self) -> Vec<[i32; 3]>
    {
        use self::config::SampleType::*;
        let ax = |k: usize| match *self {
            Centered(arr) => -(arr[k] as i32)..arr[k] as i32+1,
            Plain(arr) => 0..arr[k] as i32,
        };

        iproduct!(ax(0), ax(1), ax(2)).map(|(i,j,k)| [i,j,k]).collect()
    }

    fn points(&self, lattice: &[[f64; 3]; 3]) -> Vec<[f64; 3]>
    {
        self.signed_indices().into_iter()
            .map(|a| map_arr(a, f64::from))
            .map(|v| dot(&v, lattice))
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
pub struct ScMatrix {
    matrix: [[i32; 3]; 3],
    periods: [u32; 3],
}

impl ScMatrix {
    pub fn new(matrix: &[[i32; 3]; 3], periods: &[u32; 3]) -> Self
    {
        use ::rsp2_array_utils::det;
        // sanity check
        // (NOTE: this condition is neccessary, but not sufficient)
        assert_eq!(det(matrix).abs() as u32, periods[0] * periods[1] * periods[2]);
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
pub fn unfold_phonon(
    config: &Config,
    // Takes CoordStructure because I think there might be a case for
    // supporting <M: Eq + Hash>, with the semantics that atoms with
    // non-equal metadata are "distinct" and contributions between
    // them to a projection cannot cancel.
    superstructure: &CoordStructure,
    eigenvector_q: &SuperFracQ,
    eigenvector: KetRef,
    supercell_matrix: &ScMatrix,
) -> Vec<([u32; 3], f64)>
{
    let unfolder = UnfolderAtQ::from_config(config, superstructure, supercell_matrix, eigenvector_q);
    let indices = unfolder.q_indices().iter().cloned();
    let probs = unfolder.unfold_phonon(eigenvector);
    izip!(indices, probs).collect()
}

/// Contains precomputed information derived from the
/// q-points at which integration will be performed;
///
/// This speeds up the unfolding of many eigenvectors computed
/// at the same q point.
pub struct UnfolderAtQ {
    /// `[sc_recip_index] -> index` (3D index of sc reciprocal lattice point)
    sc_indices: Vec<[u32; 3]>,
    /// `[pc_recip_index][sc_recip_index] -> q_ket`
    q_kets_by_pc_sc: Vec<Vec<Ket>>,

    // HACK: these are only here to service the code that uses UnfolderAtQ
    /// `[sc_recip_index] -> q` (reduced into primitive cell)
    sc_qs_frac: Vec<PrimFracQ>,
    ev_q_frac: PrimFracQ,
    pc_recip_lattice: [[f64; 3]; 3],
}

impl UnfolderAtQ {
    pub fn from_config(
        config: &Config,
        superstructure: &CoordStructure,
        supercell_matrix: &ScMatrix,
        eigenvector_q: &[f64; 3], // reduced by sc lattice
    ) -> UnfolderAtQ
    {
        use ::rsp2_array_utils::{inv, map_mat, mat_from_fn, arr_from_fn};

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
                let pc_lattice = dot(&inv(&map_mat(supercell_matrix.matrix, |x| x as f64)), sc_lattice);
                let pc_inverse = inv(&pc_lattice);
                let sc_recip = mat_from_fn(|r, c| sc_inverse[c][r]);
                let pc_recip = mat_from_fn(|r, c| pc_inverse[c][r]);

                // lattice points of interest
                let sc_periods = supercell_matrix.periods;

                let quotient_sample_spec = self::config::SampleType::Plain(sc_periods);
                let quotient_indices: Vec<_> =
                    quotient_sample_spec.signed_indices()
                        .into_iter().map(|v| {
                        arr_from_fn(|k| (v[k] + sc_periods[k] as i32) as u32 % sc_periods[k])
                    }).collect();
                assert!(quotient_indices.len() > 0, "no points to sample against");

                let quotient_vecs = quotient_sample_spec.points(&sc_recip);
                let pc_recip_vecs = config.sampling.points(&pc_recip);

                // into recip cartesian space
                let eigenvector_q_cart = dot(eigenvector_q, &sc_recip);
                if eigenvector_q.iter().any(|&x| x != 0.0) {
                    // (I currently always run this code on gamma eigenvectors...)
                    warn!("Untested code path: 9fc15058-7199-45d2-80ec-630ceb575d3d");
                }

                UnfolderAtQ {
                    sc_indices: quotient_indices,
                    q_kets_by_pc_sc: pc_recip_vecs.iter().map(|sample_q| {
                        quotient_vecs.iter().map(|quotient_q| {
                            q_ket(
                                &superstructure.to_carts(),
                                &arr_from_fn(|k| eigenvector_q_cart[k] + sample_q[k] + quotient_q[k]),
                            )
                        }).collect()
                    }).collect(),
                    sc_qs_frac: {
                        let pc_recip = Lattice::new(&pc_recip);
                        Coords::Carts(quotient_vecs.clone()).to_fracs(&pc_recip)
                    },
                    pc_recip_lattice: pc_recip,
                    ev_q_frac: {
                        use ::rsp2_array_utils::dot;
                        let q = eigenvector_q_cart;
                        let q = dot(&q, &inv(&pc_recip));
                        q
                    }
                }
            },
        }
    }

    pub fn q_indices(&self) -> &[[u32; 3]]
    { &self.sc_indices }

    pub fn q_fracs(&self) -> &[[f64; 3]]
    { &self.sc_qs_frac }

    pub fn lookup_q_index(&self, q: &PrimFracQ, tol: f64) -> Option<usize>
    {
        use ::rsp2_array_utils::arr_from_fn;

        debug!("EVQ: {:?}", self.ev_q_frac);
        for q in &self.sc_qs_frac {
            debug!("QQQ: {:?}", q);
        }

        let dq: [_; 3] = arr_from_fn(|k| {
            let mut d = q[k] - self.ev_q_frac[k];
            d -= d.floor();
            d -= d.floor(); // for -EPSILON < d < 0
            d
        });

        let mut sc_qs_frac = self.sc_qs_frac.to_vec();
        for row in &mut sc_qs_frac {
            for k in 0..3 {
                row[k] -= dq[k];
            }
        }

        let sc_qs_cart =
            Coords::Fracs(sc_qs_frac.to_vec())
                .into_carts(&Lattice::new(&self.pc_recip_lattice));


        index_of_shortest_with_nearest_27(&sc_qs_cart, &self.pc_recip_lattice, tol)
    }

    pub fn unfold_phonon(&self, eigenvector: KetRef) -> Vec<f64>
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
        let mut probs = vec![0.0; num_sc_recip_vecs];
        for q_kets_by_sc in &self.q_kets_by_pc_sc {
            assert_eq!(probs.len(), q_kets_by_sc.len());
            for (prob, q_ket) in izip!(&mut probs, q_kets_by_sc) {
                for k in 0..3 {
                    *prob += q_ket.as_ref().overlap(&axis_evs[k]);
                }
            }
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
fn q_ket(carts: &[[f64; 3]], q: &[f64; 3]) -> Ket
{ carts.iter().map(|x| Rect::from_phase(-dot(x, q) * 2.0 * PI)).collect() }

type SuperFracQ = [f64; 3];
type PrimFracQ = [f64; 3];

/// Shortest vector in consideration of the 27 images around them.
// FIXME this shouldn't be just some util function, it is a common mathematical problem and
//   there are nontrivial criteria that must be met for this function to even be useful.
//  (e.g. the given cell must be reduced enough for the 27 images to cover the voronoi cell)
//  (It also could make much better use of SIMD, but that's for another time)
fn index_of_shortest_with_nearest_27(
    carts: &[[f64; 3]],
    lattice: &[[f64; 3]; 3],
    tol: f64,
) -> Option<usize>
{
    use ::rsp2_array_utils::arr_from_fn;

    assert!(carts.len() > 0);
    let mut all_carts = carts.to_vec();
    for row in lattice {
        let mut new_carts = Vec::with_capacity(all_carts.len() * 3);
        new_carts.extend(all_carts.iter().cloned());
        new_carts.extend(all_carts.iter().map(|v| arr_from_fn::<[_; 3], _>(|k| v[k] - row[k])));
        new_carts.extend(all_carts.iter().map(|v| arr_from_fn::<[_; 3], _>(|k| v[k] + row[k])));
        all_carts = new_carts;
    }

    ::util::index_of_shortest(&all_carts[..], tol)
        .map(|i| i % carts.len()) // of original, not of image
}

#[cfg(test)]
#[deny(dead_code)]
mod tests {
    use super::*;
    use ::slice_of_array::prelude::*;
    use ::rsp2_structure::{Coords, Lattice};

    #[test]
    fn simple_unfold()
    {
        const GAMMA: [f64; 3] = [0.0, 0.0, 0.0];

        fn do_it(
            structure: &CoordStructure,
            sc_vec: [i32; 3],
            expect_index: &[[u32; 3]],
            eigenvector: Vec<[f64; 3]>,
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
                &[[sc_vec[0], 0, 0], [0, sc_vec[1], 0], [0, 0, sc_vec[2]]],
                &map_arr(sc_vec, |x| x as u32),
            );
            for config in &configs {
                let unfolded = unfold_phonon(config, structure, &GAMMA, eigenvector.as_ref(), &sc_mat);

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
                    "{:?} {:?}", expect_index, unfolded);
            }
        };

        //--------------------------------------------
        // easy 1D case
        // 1 atom per primitive cell
        let structure = CoordStructure::new_coords(
            Lattice::diagonal(&[1.0, 1.0, 4.0]),
            Coords::Carts(vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ]),
        );
        let sc_vec = [1, 1, 4];
        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        go_do_it(&[[0, 0, 0]], vec![
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ]);

        // phase rotation of 2/4 tau (i.e. `-1`) per unit cell
        go_do_it(&[[0, 0, 2]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
        ]);

        // phase rotation of 1/4 tau (i.e. `i`) per unit cell.
        // because gamma eigenvectors are always real, we end up
        //  with contributions from two kpoints whose imaginary
        //  parts cancel.
        go_do_it(&[[0, 0, 1], [0, 0, 3]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ]);

        //--------------------------------------------
        // supercell along multiple dimensions
        // 1 atom per primitive cell
        let structure = CoordStructure::new_coords(
            Lattice::diagonal(&[2.0, 2.0, 1.0]),
            Coords::Carts(vec![
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]),
        );
        let sc_vec = [2, 2, 1];

        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        // gamma
        let p = 8_f64.sqrt().recip();
        go_do_it(&[[0, 0, 0]], vec![
            [ p,  p, 0.0],
            [ p,  p, 0.0],
            [ p,  p, 0.0],
            [ p,  p, 0.0],
        ]);

        // non-gamma along one axis
        let p = 8_f64.sqrt().recip();
        go_do_it(&[[0, 1, 0]], vec![
            [-p,  p, 0.0],
            [ p, -p, 0.0],
            [-p,  p, 0.0],
            [ p, -p, 0.0],
        ]);

        // non-gamma along multiple axis.
        let p = 8_f64.sqrt().recip();
        go_do_it(&[[1, 1, 0]], vec![
            [-p,  p, 0.0],
            [ p, -p, 0.0],
            [ p, -p, 0.0],
            [-p,  p, 0.0],
        ]);

        //--------------------------------------------
        // non-diagonal lattice.
        // hopefully, this will catch bugs involving incorrect
        //   usage of matrices vs their transpose.
        // 1 atom per primitive cell
        let structure = CoordStructure::new_coords(
            Lattice::new(&[
                [1.0, 0.0, 0.0],
                [-0.5, 0.5 * 3_f64.sqrt(), 0.0],
                [0.0, 0.0, 1.0],
            ]),
            Coords::Fracs(vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ]),
        );
        let sc_vec = [2, 2, 1];

        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        go_do_it(&[[1, 1, 0]], vec![
            [0.0, 0.0, -0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
        ]);

        //--------------------------------------------
        // primitive structure with more than one atom
        let structure = CoordStructure::new_coords(
            Lattice::new(&[
                // graphene cell (2 atoms per primitive),
                // doubled along b (4 atoms per supercell)
                [1.0, 0.0, 0.0],
                [-1.0, 1.0 * 3_f64.sqrt(), 0.0],
                [0.0, 0.0, 1.0],
            ]),
            Coords::Fracs(vec![
                // honeycomb pattern
                [    0.0, 0.0, 0.0],
                [1.0/3.0, 0.0, 0.0],
                [    0.0, 0.5, 0.0],
                [1.0/3.0, 0.5, 0.0],
            ]),
        );
        let sc_vec = [1, 2, 1];

        let go_do_it = |expected, eigenvector|
            do_it(&structure, sc_vec, expected, eigenvector);

        // the obvious gamma vec
        go_do_it(&[[0, 0, 0]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
        ]);

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
        ]);

        // a non-gamma vec
        go_do_it(&[[0, 1, 0]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ]);

        //--------------------------------------------

        // TODO: Test with non-perfect supercell
        // TODO: Test with eigenvector_q not at gamma
        // TODO: Test with non-diagonal supercell
    }
}