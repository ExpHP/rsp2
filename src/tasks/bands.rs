
use ::rsp2_structure::Structure;
use ::rsp2_array_utils::{vec_from_fn, dot};
use ::rsp2_kets::{Ket, KetRef, Rect};
use ::slice_of_array::prelude::*;

use ::std::f64::consts::PI;
use ::std::hash::Hash;
use ::std::collections::HashMap;

pub use self::config::Config;
pub mod config {
    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[serde(rename_all = "kebab-case")]
    pub struct Config {
        pub fbz: FbzType,
        pub sampling: IntegralType,
    }

    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[derive(Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    #[serde(rename_all = "kebab-case")]
    pub enum FbzType {
        /// Parallelepiped in reciprocal space.
        Approximate,

        /// Voronoi cell in reciprocal space.
        ///
        /// As the name suggests, this is what you should want to use...
        /// if it were implemented.  Which it isn't.  Yet.
        Correct,
    }

    /// How to perform an integration over an FBZ.
    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Clone)]
    #[derive(PartialEq, Eq, Hash)]
    #[serde(rename_all = "kebab-case")]
    pub enum IntegralType {
        /// Only sample the gamma point.
        Gamma,
        // there could be things like tetrahedron method, and monkhurst-pack...
    }
}

pub fn unfold_phonon<M: Eq + Hash + Clone>(
    config: &Config,
    superstructure: &Structure<M>,
    supercell_q: &SuperFracQ,
    eigenvector: KetRef,
    supercell_matrix: &[[i32; 3]; 3],
) -> Vec<([u32; 3], f64)>
{
    assert_eq!(eigenvector.len(), 3 * superstructure.num_atoms());
    for (i, j) in vec![(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)] {
        assert_eq!(supercell_matrix[i][j], 0, "non-diagonal supercells not supported (yet)")
    }

    let sc_dim = vec_from_fn(|k| {
        assert!(supercell_matrix[k][k] > 0, "negative supercell not supported (yet)");
        supercell_matrix[k][k] as u32
    });

    let quotient_data = approximate_fbz_samples(&sc_dim, &config.sampling);
    let direct_sum = make_direct_sum(superstructure, eigenvector);

    // for each Q in the quotient space of the two FBZs
    quotient_data.iter().map(|&(quotient_q_index, ref ks_and_weights)| {

        // integrate over the image of the supercell FBZ that corresponds to this Q
        let total = ks_and_weights.iter().map(|&(ref q_unit_frac, q_weight)| {
            let q_cart = dot(superstructure.lattice().inverse_matrix(), q_unit_frac);

            // sum over distinguishable components of the eigenvector
            // (e.g. polarized excitations of a specific element along a specific axis)
            direct_sum.iter().map(|&DirectSumItem { ref carts, ref eigenvector }| {
                // project this component onto the q-point of integration.
                // terms may cancel here...
                izip!(carts, eigenvector.as_ref())
                    .map(|(cart, ev_elem)| Rect::from_phase(-dot(cart, &q_cart) * 2.0 * PI) * ev_elem)
                    .fold(Rect::zero(), |a, b| a + b)
                    .sqnorm()
            }).sum::<f64>() * q_weight
        }).sum::<f64>();

        (quotient_q_index, total / quotient_data.len() as f64)
    }).collect()
}


// WeightMap[i] = (output, [(input, weight) that contribute])
type WeightMap<K, V> = Vec<(V, Vec<(K, f64)>)>;
type SuperFracQ = [f64; 3];
type UnitFracQ = [f64; 3];
type CartQ = [f64; 3];

// output has "floatly-typed" integers
fn fbz_samples(
    sc_dim: &[u32; 3],
    fbz: config::FbzType,
    sampling: &config::IntegralType,
) -> WeightMap<SuperFracQ, [u32; 3]>
{
    use self::config::FbzType::*;
    match fbz {
        Approximate => approximate_fbz_samples(sc_dim, sampling),
        Correct => panic!("correct form of FBZ not yet implemented (sorry)"),
    }
}

// output has "floatly-typed" integers.
// This uses a parallelepiped FBZ which is, strictly speaking, not correct.
fn approximate_fbz_samples(
    sc_dim: &[u32; 3],
    // how integration is performed over each image of the supercell FBZ
    sampling: &config::IntegralType,
) -> WeightMap<SuperFracQ, [u32; 3]>
{
    // FIXME
    // even n gets even trickier with other types of sampling because
    // some of the points may get distributed to opposite sides of the FBZ.
    // In general this seems too hard to just code manually; we need
    // a higher level abstraction for constructing these weighted integrations.

    match *sampling {
        config::IntegralType::Gamma => {
            let axis_inds: [Vec<Vec<(i32, f64)>>; 3] = vec_from_fn(|k| {
                let d = (sc_dim[k] / 2) as i32;
                let r = sc_dim[k] % 2;

                match r {
                    // odd;
                    // each image -d, -d+1, ..., d-1, d is distinct
                    1 => (0..d+1).chain(-d..0).map(|i| vec![(i, 1.0)]).collect(),

                    // even;
                    // one of the images of gamma lies on the FBZ boundary
                    // and must be replaced with a sum
                    0 => {
                        let mut out: Vec<_> = (0..d).map(|i| vec![(i, 1.0)]).collect();
                        out.push(vec![(d, 0.5), (-d, 0.5)]);
                        out.extend((-d+1..0).map(|i| vec![(i, 1.0)]));
                        out
                    },

                    _ => unreachable!(),
                }
            });

            // Behold, The Great Pyramid
            let mut out = vec![];
            for (a_i, a_list) in axis_inds[0].iter().enumerate() {
                for (b_i, b_list) in axis_inds[1].iter().enumerate() {
                    for (c_i, c_list) in axis_inds[2].iter().enumerate() {

                        let index = [a_i as u32, b_i as u32, c_i as u32];
                        let mut item = vec![];
                        for &(a, wa) in a_list {
                            for &(b, wb) in b_list {
                                for &(c, wc) in c_list {
                                    let q = [a as f64, b as f64, c as f64];
                                    let weight = wa * wb * wc;
                                    item.push((q, weight))
                                }
                            }
                        }
                        out.push((index, item));
                    }
                }
            }
            out
        },
    }
}

/// Decompose the eigenvector into a direct sum over
/// distinguishable species and cartesian axes
fn make_direct_sum<M: Hash + Eq>(
    structure: &Structure<M>,
    eigenvector: KetRef,
) -> Vec<DirectSumItem>
{
    assert_eq!(eigenvector.len(), 3 * structure.num_atoms());
    let metas = structure.metadata();
    let mut map = HashMap::new();
    for (cart, m, complex) in izip!(structure.to_carts(), metas, eigenvector) {
        for k in 0..3 {
            let key = (m, k);
            map.entry(key).or_insert_with(|| vec![]).push((cart, complex));
        }
    }

    map.into_iter().map(|(_, data)| {
        let (carts, complex): (_, Vec<_>) = data.into_iter().unzip();
        let eigenvector = complex.into_iter().collect();
        DirectSumItem { carts, eigenvector }
    }).collect()
}

// a subset of the eigenvector whose elements are able
//  to cancel in the projection
struct DirectSumItem {
    carts: Vec<[f64; 3]>,
    eigenvector: Ket,
}

mod tests {
    use super::*;

    #[test]
    fn test_approximate_fbz_samples()
    {
        use super::config::IntegralType;

        fn sorted_call(sc_dim: &[u32; 3], sampling: IntegralType)
        -> WeightMap<[f64; 3], [u32; 3]>
        {
            let mut v = approximate_fbz_samples(&sc_dim, &sampling);
            for &mut (_, ref mut el) in &mut v {
                el.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };

        // is our hair not on fire
        assert_eq!(
            sorted_call(&[1, 1, 1], IntegralType::Gamma),
            vec![
                ([0, 0, 0], vec![([0.0, 0.0, 0.0], 1.0)]),
            ]);

        // even dim
        assert_eq!(
            sorted_call(&[1, 2, 1], IntegralType::Gamma),
            vec![
                ([0, 0, 0], vec![([0.0,  0.0, 0.0], 1.0)]),
                ([0, 1, 0], vec![
                    ([0.0, -1.0, 0.0], 0.5),
                    ([0.0,  1.0, 0.0], 0.5),
                ]),
            ]);

        // odd dim
        assert_eq!(
            sorted_call(&[1, 3, 1], IntegralType::Gamma),
            vec![
                ([0, 0, 0], vec![([0.0,  0.0, 0.0], 1.0)]),
                ([0, 1, 0], vec![([0.0,  1.0, 0.0], 1.0)]),
                ([0, 2, 0], vec![([0.0, -1.0, 0.0], 1.0)]),
            ]);

        // multiple nontrivial dims
        assert_eq!(
            sorted_call(&[1, 2, 2], IntegralType::Gamma),
            vec![
                ([0, 0, 0], vec![
                    ([0.0,  0.0, 0.0], 1.0),
                ]),
                ([0, 0, 1], vec![
                    ([0.0, 0.0, -1.0], 0.5),
                    ([0.0, 0.0,  1.0], 0.5),
                ]),
                ([0, 1, 0], vec![
                    ([0.0, -1.0, 0.0], 0.5),
                    ([0.0,  1.0, 0.0], 0.5),
                ]),
                ([0, 1, 1], vec![
                    ([0.0, -1.0, -1.0], 0.25),
                    ([0.0, -1.0,  1.0], 0.25),
                    ([0.0,  1.0, -1.0], 0.25),
                    ([0.0,  1.0,  1.0], 0.25),
                ]),
            ]);
    }
}
