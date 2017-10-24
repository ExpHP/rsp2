//! Test case for algorithms on finite groups.
//!
//! We consider the following representations of the octohedral group:
//!
//! * Signed permutations of the XYZ axes
//! * Permutations of the 8 vertices of the cube on `[-1, 1]^3`
//! * Rotation matrices

use ::rsp2_array_utils::{dot, vec_from_fn, mat_from_fn};
use ::util::perm::Perm;
use ::FracRot;

/// Uniquely represents an element of the octahedral group
/// as a signed permutation of the XYZ axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SignedPerm {
    signs: [i32; 3],
    perm: [i32; 3],
}

impl SignedPerm {
    fn from_rot(rot: &FracRot) -> Self
    {
        let m_t = rot.float_t();
        let m: [[_; 3]; 3] = mat_from_fn(|r, c| m_t[c][r]);

        // (stupid hat trick...)
        let v: [f64; 3] = dot(&m, &[0.0, 1.0, 2.0]);
        SignedPerm {
            signs: vec_from_fn(|k| v[k].signum() as i32),
            perm: vec_from_fn(|k| v[k].abs() as i32),
        }
    }

    fn to_rot(&self) -> FracRot
    {
        let mut m = [[0; 3]; 3];
        for k in 0..3 {
            m[k][self.perm[k] as usize] = self.signs[k];
        }
        FracRot::new(&m)
    }
}

// ascribes an index to each vertex of the cube whose vertices
// lie on `{-1, 1}^3`.
fn index_from_point(point: [i32; 3]) -> u32
{
    assert!(point.iter().all(|x| x.abs() == 1));
    let (a, b, c) = tup3(point);
    // binary encoding of bits `k => (point[k] + 1) / 2`
    ((4 * (a + 1) + 2 * (b + 1) + (c + 1)) / 2) as u32
}

// inverse of `index_from_point`
fn point_from_index(index: u32) -> [i32; 3]
{
    let index = index as i32;
    let c = ((index / 1) % 2) * 2 - 1;
    let b = ((index / 2) % 2) * 2 - 1;
    let a = ((index / 4) % 2) * 2 - 1;
    [a, b, c]
}

fn vertex_perm_from_rot(rot: &FracRot) -> Perm
{
    let points: Vec<_> =
        (0..8).map(point_from_index)
        .map(|v| vec_from_fn(|k| f64::from(v[k])))
        .collect();

    let points = rot.transform_prim(&points);

    let indices =
        points.into_iter()
        .map(|v| vec_from_fn(|k| v[k] as i32))
        .map(index_from_point).collect();

    Perm::from_vec(indices).unwrap()
}

fn tup3<T: Clone>(a: [T; 3]) -> (T, T, T)
{ (a[0].clone(), a[1].clone(), a[2].clone()) }

const N_GENERATORS: usize = 3;
lazy_static! {
    static ref SIGNED_PERM_GENERATORS: [SignedPerm; N_GENERATORS] = {
        [
            SignedPerm { signs: [ 1, 1, 1], perm: [1, 2, 0] }, // x -> y -> z
            SignedPerm { signs: [ 1, 1, 1], perm: [1, 0, 2] }, // x <-> y
            SignedPerm { signs: [-1, 1, 1], perm: [0, 1, 2] }, // mirror x
        ]
    };

    static ref VERTEX_PERM_GENERATORS: [Perm; N_GENERATORS] = {
        [
            Perm::from_vec(vec![0, 2, 4, 6, 1, 3, 5, 7]).unwrap(), // x -> y -> z
            Perm::from_vec(vec![0, 1, 4, 5, 2, 3, 6, 7]).unwrap(), // x <-> y
            Perm::from_vec(vec![4, 5, 6, 7, 0, 1, 2, 3]).unwrap(), // mirror x
        ]
    };

    static ref ROTATION_GENERATORS: [FracRot; N_GENERATORS] = {
        [
            // x -> y -> z
            FracRot::new(&[
                [ 0, 1, 0],
                [ 0, 0, 1],
                [ 1, 0, 0],
            ]),
            // x <-> y
            FracRot::new(&[
                [ 0, 1, 0],
                [ 1, 0, 0],
                [ 0, 0, 1],
            ]),
            // mirror x
            FracRot::new(&[
                [-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 1],
            ]),
        ]
    };
}

mod tests {
    use super::*;

    fn validate_homomorphism<G, H, GFn, HFn>(
        gs: &[G], mut g_fn: GFn,
        hs: &[H], mut h_fn: HFn,
    )
    where
        H: Eq + Clone + ::std::fmt::Debug,
        G: Eq + Clone + ::std::hash::Hash,
        GFn: FnMut(&G, &G) -> G,
        HFn: FnMut(&H, &H) -> H,
    {
        use ::std::collections::HashMap;

        let h_by_g: HashMap<_,_> =
            izip!(gs.iter().cloned(), hs.iter().cloned()).collect();

        assert_eq!(gs.len(), hs.len());
        for a in 0..gs.len() {
            for b in 0..gs.len() {
                let g = g_fn(&gs[a], &gs[b]);
                let h = h_fn(&hs[a], &hs[b]);
                assert_eq!(h_by_g[&g], h);
            }
        }
    }


    fn test_generators()
    {
        for i in 0..N_GENERATORS {
            let s_perm = SIGNED_PERM_GENERATORS[i].clone();
            let v_perm = VERTEX_PERM_GENERATORS[i].clone();
            let rot = ROTATION_GENERATORS[i].clone();

            assert_eq!(s_perm, SignedPerm::from_rot(&rot));
            assert_eq!(rot, s_perm.to_rot());
            assert_eq!(v_perm, vertex_perm_from_rot(&rot))
        }
    }

    #[test]
    fn test_generated_groups()
    {
        //use ::algo::group::generate_finite_group;
        unimplemented!()
        // TODO:
        // * use generate_finite_group on all three sets of generators
        // * check that each one has 48 elements
        // * perform comparisons from test_generators,
        //   and validate homomorphisms in both directions between pairs
        //   that don't have conversions


        // s_perms = generate_finite_group(SIGNED_PERM_GENERATORS, /* TODO */)
    }

}
