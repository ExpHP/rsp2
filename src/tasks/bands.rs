
use ::rsp2_structure::Structure;
use ::rsp2_array_utils::{arr_from_fn, dot};
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
        /// Evenly-spaced distribution around gamma.
        GammaCentered([u32; 3]),
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

    let sc_dim = arr_from_fn(|k| {
        assert!(supercell_matrix[k][k] > 0, "negative supercell not supported (yet)");
        supercell_matrix[k][k] as u32
    });

    let quotient_data = approximate_fbz_samples(&sc_dim, &config.sampling);
    let direct_sum = make_direct_sum(superstructure, eigenvector);

    // for each Q in the quotient space of the two FBZs
    let mut out: Vec<_> = quotient_data.iter().map(|&(quotient_q_index, ref ks_and_weights)| {

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

        // the test bra has a norm of sqrt(N); cancel that out
        (quotient_q_index, total / superstructure.num_atoms() as f64)
    }).collect();

    // HACK:
    let total: f64 = out.iter().map(|&(_, p)| p).sum();
    for &mut (_, ref mut p) in &mut out {
        *p /= total;
    }
    out
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
    match *sampling {
        config::IntegralType::GammaCentered(k_divs) => {

            let by_axis: [_; 3] =
                arr_from_fn(|k| gamma_centered_1d_kdivs(sc_dim[k], k_divs[k]));

            // Behold, The Great Pyramid
            let mut out = vec![];
            for &(a_i, ref a_list) in by_axis[0].iter() {
                for &(b_i, ref b_list) in by_axis[1].iter() {
                    for &(c_i, ref c_list) in by_axis[2].iter() {

                        let index = [a_i, b_i, c_i];
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

// FIXME The implementation here is so retardedly complex,
//       and I just don't know how to write it any nicer.
//       (And this is *without* even taking the voronoi cell into account!!)
//
//       I believe (but have not confirmed) that, for our goal of projecting
//       onto supercell BZs, we really DO need to make sure that each sample
//       point is associated with the correct (closest) image of the supercell
//       gamma (we cannot just use points in the half-open interval 0..1 and
//       call it a day).  We also cannot just generate points and map them
//       into [-0.5, 0.5] because points that land precisely on boundaries
//       need to have their contributions split up.
fn gamma_centered_1d_kdivs(
    sc_dim: u32,
    k_divs: u32,
) -> WeightMap<f64, u32>
{
    // FIXME
    // even k-divs cause one of the k-points in each BZ
    // to be split across the boundary.
    // This is on top of the already existing edge case of
    // even numbers in the *supercell*.
    assert!(k_divs % 2 == 1,
        "even numbers with Gamma-centered not supported (yet)");

    // in the common case there will be k_divs[k] equally weighted
    //  points with offsets of  -0.5 < x < 0.5.  However, if sc_dims[k]
    //  is even then one of the BZs is split across the boundary.
    //  Worse yet, the split occurs directly on top of that BZ's gamma.
    //
    // encode this possible splitting into two sets of weights which may or may
    // not be used from the same reference point.
    let (lows, highs, easys) = {
        use ::std::iter::once;
        let m = (k_divs - 1) as i32 / 2;

        // split case
        let lows: Vec<_> = ichain!(
            once((0.0, 0.5)),
            (-m..-1 + 1).map(|x| (x as f64 / k_divs as f64, 1.0)),
            // (-m..-1 + 1).map(|x| (1.0 / x as f64, 1.0)),
        ).collect();

        let highs: Vec<_> = ichain!(
            ( 1..m + 1).map(|x| (x as f64 / k_divs as f64, 1.0)),
            // ( 1..m + 1).map(|x| (1.0 / x as f64, 1.0)),
            once((0.0, 0.5)),
        ).collect();

        // contiguous case
        let easys: Vec<_> = ichain!(
            (-m..m + 1).map(|x| (x as f64 / k_divs as f64, 1.0)),
            // (-m..m + 1).map(|x| (match x { 0 => 0.0, x => 1.0 / x as f64 }, 1.0)),
        ).collect();

        (lows, highs, easys)
    };

    // contiguous BZs are those with gammas located at
    //  -easy_max, -easy_max+1, ..., easy_max.
    // This expression was simply worked out by case analysis.
    let easy_max = ((sc_dim / 2) + (sc_dim % 2)) as i32 - 1;

    let mut out = vec![];
    // origin and BZs that lie entirely right of it
    for n in 0..easy_max + 1 {
        out.push(
            easys.iter().map(|&(off, w)| (n as f64 + off, w)).collect()
        );
    }

    match sc_dim % 2 {
        // even supercell; handle the split BZ
        0 => {
            let d = (sc_dim / 2) as f64;
            out.push(ichain!(
                lows.iter().map(|&(off, w)| (d + off, w)),
                highs.iter().map(|&(off, w)| (-d + off, w)),
            ).collect());
        },
        1 => { },
        _ => unreachable!(),
    }

    // BZs entirely left of the origin
    for n in -easy_max..-1 + 1 {
        out.push(
            easys.iter().map(|&(off, w)| (n as f64 + off, w)).collect()
        )
    }

    // (add indices for WeightMap)
    out.into_iter().enumerate().map(|(i, val)| (i as u32, val)).collect()
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

    // HACK
    let eigenvector = eigenvector.iter().collect::<Vec<_>>().nest::<[_; 3]>().to_vec();

    for (cart, m, complex) in izip!(structure.to_carts(), metas, eigenvector) {
        for k in 0..3 {
            let key = (m, k);
            map.entry(key).or_insert_with(|| vec![]).push((cart, complex[k]));
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

#[cfg(test)]
#[deny(dead_code)]
mod tests {
    use super::*;
    use rsp2_structure::{Coords, Lattice};
    use ::std::fmt::Debug;

    // trait for tests of absolute tolerance
    trait AlmostEq: Sized {
        fn almost_eq(tol: f64, a: &Self, b: &Self) -> bool;
    }

    impl AlmostEq for f64 {
        fn almost_eq(tol: f64, a: &Self, b: &Self) -> bool
        { (a - b).abs() <= tol }
    }

    impl AlmostEq for [f64; 3] {
        fn almost_eq(tol: f64, a: &Self, b: &Self) -> bool
        { izip!(a, b).all(|(a, b)| AlmostEq::almost_eq(tol, a, b)) }
    }

    fn assert_weight_maps_almost_eq<K:, V: >(
        tol: f64,
        mut a: WeightMap<K, V>,
        mut b: WeightMap<K, V>,
    ) where
        K: Debug + AlmostEq + PartialOrd,
        V: Debug + Ord,
    {
        a.sort_by(|x, y| x.0.cmp(&y.0));
        b.sort_by(|x, y| x.0.cmp(&y.0));
        for x in &mut a { x.1.sort_by(|x, y| x.partial_cmp(y).unwrap()) }
        for x in &mut b { x.1.sort_by(|x, y| x.partial_cmp(y).unwrap()) }

        let (full_a, full_b) = (a, b);
        for (ref a, ref b) in izip!(&full_a, &full_b) {
            assert_eq!(a.0, b.0, "\n{:?}\n{:?}", full_a, full_b);
            for (a, b) in izip!(&a.1, &b.1) {
                assert!(
                    AlmostEq::almost_eq(tol, &a.0, &b.0)
                    && AlmostEq::almost_eq(tol, &a.1, &b.1),
                    "\n{:?}\n{:?}", full_a, full_b);
            }
        }
    }

    #[test]
    fn test_approximate_fbz_samples()
    {
        use super::config::IntegralType::{GammaCentered};

        // is our hair not on fire
        assert_weight_maps_almost_eq(0.0,
            approximate_fbz_samples(&[1, 1, 1], &GammaCentered([1, 1, 1])),
            vec![
                ([0, 0, 0], vec![([0.0, 0.0, 0.0], 1.0)]),
            ]);

        // even dim
        assert_weight_maps_almost_eq(0.0,
            approximate_fbz_samples(&[1, 2, 1], &GammaCentered([1, 1, 1])),
            vec![
                ([0, 0, 0], vec![([0.0,  0.0, 0.0], 1.0)]),
                ([0, 1, 0], vec![
                    ([0.0, -1.0, 0.0], 0.5),
                    ([0.0,  1.0, 0.0], 0.5),
                ]),
            ]);

        // odd dim
        assert_weight_maps_almost_eq(0.0,
            approximate_fbz_samples(&[1, 3, 1], &GammaCentered([1, 1, 1])),
            vec![
                ([0, 0, 0], vec![([0.0,  0.0, 0.0], 1.0)]),
                ([0, 1, 0], vec![([0.0,  1.0, 0.0], 1.0)]),
                ([0, 2, 0], vec![([0.0, -1.0, 0.0], 1.0)]),
            ]);

        // multiple nontrivial dims
        assert_weight_maps_almost_eq(0.0,
            approximate_fbz_samples(&[1, 2, 2], &GammaCentered([1, 1, 1])),
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

        // odd kdivs, odd supercell
        let t = 1./3.;
        assert_weight_maps_almost_eq(1e-11,
            approximate_fbz_samples(&[1, 1, 3], &GammaCentered([1, 1, 3])),
            vec![
                ([0, 0, 0], vec![
                    ([0.0, 0.0, 0.0 -   t], 1.0),
                    ([0.0, 0.0, 0.0 + 0.0], 1.0),
                    ([0.0, 0.0, 0.0 +   t], 1.0),
                ]),
                ([0, 0, 1], vec![
                    ([0.0, 0.0, 1.0 -   t], 1.0),
                    ([0.0, 0.0, 1.0 + 0.0], 1.0),
                    ([0.0, 0.0, 1.0 +   t], 1.0),
                ]),
                ([0, 0, 2], vec![
                    ([0.0, 0.0, -1.0 -   t], 1.0),
                    ([0.0, 0.0, -1.0 + 0.0], 1.0),
                    ([0.0, 0.0, -1.0 +   t], 1.0),
                ]),
            ]);

        // odd kdivs, even supercell
        let t = 1./3.;
        assert_weight_maps_almost_eq(1e-11,
            // NOTE: a supercell of 2 is not large enough to exercise all
            //        of the code paths that add points to the output.
            approximate_fbz_samples(&[1, 1, 4], &GammaCentered([1, 1, 3])),
            vec![
                ([0, 0, 0], vec![
                    ([0.0, 0.0,  0.0 -   t], 1.0),
                    ([0.0, 0.0,  0.0 + 0.0], 1.0),
                    ([0.0, 0.0,  0.0 +   t], 1.0),
                ]),
                ([0, 0, 1], vec![
                    ([0.0, 0.0,  1.0 -   t], 1.0),
                    ([0.0, 0.0,  1.0 + 0.0], 1.0),
                    ([0.0, 0.0,  1.0 +   t], 1.0),
                ]),
                ([0, 0, 2], vec![
                    ([0.0, 0.0, -2.0 + 0.0], 0.5),
                    ([0.0, 0.0, -2.0 +   t], 1.0),
                    ([0.0, 0.0,  2.0 -   t], 1.0),
                    ([0.0, 0.0,  2.0 - 0.0], 0.5),
                ]),
                ([0, 0, 3], vec![
                    ([0.0, 0.0, -1.0 -   t], 1.0),
                    ([0.0, 0.0, -1.0 + 0.0], 1.0),
                    ([0.0, 0.0, -1.0 +   t], 1.0),
                ]),
            ]);
    }

    #[cfg(nope)] // FIXME
    #[test]
    fn simple_unfold()
    {
        const GAMMA: [f64; 3] = [0.0, 0.0, 0.0];

        fn do_it(structure: &Structure<()>, sc_vec: [i32; 3], expect_index: &[[u32; 3]], eigenvector: Vec<[f64; 3]>) {
            let config = from_json!({
                "fbz": "approximate",
                "sampling": { "gamma-centered": [9, 9, 9] },
            });
            let eigenvector: Ket = eigenvector.flat().iter().map(|&r| Rect::from(r)).collect();
            let sc_mat = [[sc_vec[0], 0, 0], [0, sc_vec[1], 0], [0, 0, sc_vec[2]]];
            let unfolded = unfold_phonon(&config, structure, &GAMMA, eigenvector.as_ref(), &sc_mat);

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
        };

        //--------------------------------------------
        // easy 1D case
        let structure = Structure::new_coords(
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

        go_do_it(&[[0, 0, 2]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
        ]);

        go_do_it(&[[0, 0, 1], [0, 0, 3]], vec![
            [0.0, 0.0,  0.5],
            [0.0, 0.0,  0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ]);

        //--------------------------------------------
        // supercell along multiple dimensions
        let structure = Structure::new_coords(
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

        let p = 8_f64.sqrt().recip();
        go_do_it(&[[0, 0, 0]], vec![
            [ p,  p, 0.0],
            [ p,  p, 0.0],
            [ p,  p, 0.0],
            [ p,  p, 0.0],
        ]);

        let p = 8_f64.sqrt().recip();
        go_do_it(&[[0, 1, 0]], vec![
            [-p,  p, 0.0],
            [ p, -p, 0.0],
            [-p,  p, 0.0],
            [ p, -p, 0.0],
        ]);

        let p = 8_f64.sqrt().recip();
        go_do_it(&[[1, 1, 0]], vec![
            [-p,  p, 0.0],
            [ p, -p, 0.0],
            [ p, -p, 0.0],
            [-p,  p, 0.0],
        ]);


        //--------------------------------------------
        // non-diagonal lattice
        let structure = Structure::new_coords(
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
        let structure = Structure::new_coords(
            Lattice::new(&[
                // doubled along b
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
    }

}
