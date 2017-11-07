//! Test case for algorithms on finite groups.
//!
//! This module was created to "debug" a flaw in my reasoning about
//! the relationship between rotations and permutations.  It now serves
//! as a demonstration of the peculiar properties of this relationship,
//! and a measure of protection against future mistakes of the same.
//!
//! We consider the following representations of the octohedral group:
//!
//! * Signed permutations of the XYZ axes
//! * Rotation matrices
//! * Permutations of the 8 vertices of the cube on `[-1, 1]^3`
//!
//! Some interesting facts:
//!
//! * While permutations and rotations both are groups, **they compose
//!   differently.**  This is because rotations act on values (indepent
//!   of index), while permutations act on indices (independent of value).
//! * In fact, permutations are not a suitable representation
//!   of a spacegroup G.
//! * When looking at permutations of a specific input vector of positions,
//!   these permutations are, in fact, a representation of the *opposite group*
//!   *G*<sup>op</sup>, where the operations compose in reverse.
//!   See https://github.com/ExpHP/rsp2/issues/1#issuecomment-340279243

use ::rsp2_array_utils::{dot, map_arr, mat_from_fn};
use ::{Perm, FracRot};
#[allow(unused)]
use ::Permute; // FIXME I do not know why this

/// Type of positions being acted upon.
type X = Vec<[f64; 3]>;

//-------------------------------------------------------------------
// Define an initial input vector.

lazy_static! {
    static ref INIT_X: X = (0..8).map(point_from_index).collect();
}

// ascribes an index to each vertex of the cube whose vertices
// lie on `{-1, 1}^3`.
fn index_from_point(point: [f64; 3]) -> u32
{
    assert!(point.iter().all(|x| x.abs() == 1.0));
    let (a, b, c) = tup3(point);
    // binary encoding of bits `k => (point[k] + 1) / 2`
    ((4 * (a as i32 + 1) + 2 * (b as i32 + 1) + (c as i32 + 1)) / 2) as u32
}

// inverse of `index_from_point`
fn point_from_index(index: u32) -> [f64; 3]
{
    let index = index as i32;
    let c = ((index / 1) % 2) * 2 - 1;
    let b = ((index / 2) % 2) * 2 - 1;
    let a = ((index / 4) % 2) * 2 - 1;
    [a as f64, b as f64, c as f64]
}

// interpret a rotation as a permutation on INIT_X
fn vertex_perm_from_rot(rot: &FracRot) -> Perm
{
    let points = rot.transform_prim(&INIT_X);
    let indices = points.into_iter().map(index_from_point).collect();
    Perm::from_vec(indices).unwrap()
}

fn tup3<T: Clone>(a: [T; 3]) -> (T, T, T)
{ (a[0].clone(), a[1].clone(), a[2].clone()) }

//-------------------------------------------------------------------
// A special representation unique to this group,
// just to add more redundancy to our tests.

/// Uniquely represents an element of the octahedral group
/// as a signed permutation of the XYZ axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SignedPerm {
    signs: [i32; 3],
    perm: [i32; 3],
}

impl SignedPerm {
    pub fn eye() -> Self
    { SignedPerm { signs: [1, 1, 1], perm: [0, 1, 2] } }

    pub fn from_rot(rot: &FracRot) -> Self
    {
        let m_t = rot.float_t();
        let m: [[_; 3]; 3] = mat_from_fn(|r, c| m_t[c][r]);

        // (stupid hat trick...)
        let v: [f64; 3] = dot(&m, &[1.0, 2.0, 3.0]);
        SignedPerm {
            signs: map_arr(v, |x| x.signum() as i32),
            perm: map_arr(v, |x| x.abs() as i32 - 1),
        }
    }

    pub fn to_rot(&self) -> FracRot
    {
        let mut m = [[0; 3]; 3];
        for k in 0..3 {
            m[k][self.perm[k] as usize] = self.signs[k];
        }
        FracRot::new(&m)
    }

    pub fn then(&self, next: &SignedPerm) -> SignedPerm
    {
        let mut perm = [0x0B0E; 3];
        let mut signs = [0x0B0E; 3];
        for k in 0..3 {
            perm[k] = self.perm[next.perm[k] as usize];
            signs[k] = self.signs[next.perm[k] as usize];
            signs[k] *= next.signs[k];
        }
        SignedPerm { perm, signs }
    }

    pub fn transform(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]>
    { self.to_rot().transform_prim(points) }
}

//-------------------------------------------------------------------
// Define generators in matching order between the various groups.

const N_GENERATORS: usize = 3;
lazy_static! {

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

    static ref SIGNED_PERM_GENERATORS: [SignedPerm; N_GENERATORS] = {
        [
            SignedPerm { signs: [ 1, 1, 1], perm: [1, 2, 0] }, // x -> y -> z
            SignedPerm { signs: [ 1, 1, 1], perm: [1, 0, 2] }, // x <-> y
            SignedPerm { signs: [-1, 1, 1], perm: [0, 1, 2] }, // mirror x
        ]
    };

    // NOTE: Vertex perms depend on the position vector!
    //       These assume that the input is INIT_X.
    static ref VERTEX_PERM_GENERATORS: [Perm; N_GENERATORS] = {
        [
            Perm::from_vec(vec![0, 2, 4, 6, 1, 3, 5, 7]).unwrap(), // x -> y -> z
            Perm::from_vec(vec![0, 1, 4, 5, 2, 3, 6, 7]).unwrap(), // x <-> y
            Perm::from_vec(vec![4, 5, 6, 7, 0, 1, 2, 3]).unwrap(), // mirror x
        ]
    };

    // Expected results of application to INIT_X
    static ref EXPECTED_ACTIONS: [X; N_GENERATORS] = {
        [
            // x -> y -> z
            vec![
                [-1.0, -1.0, -1.0],
                [-1.0,  1.0, -1.0],
                [ 1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [-1.0,  1.0,  1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0,  1.0],
            ],
            // x <-> y
            vec![
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [ 1.0, -1.0, -1.0],
                [ 1.0, -1.0,  1.0],
                [-1.0,  1.0, -1.0],
                [-1.0,  1.0,  1.0],
                [ 1.0,  1.0, -1.0],
                [ 1.0,  1.0,  1.0],
            ],
            // mirror x
            vec![
                [ 1.0, -1.0, -1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0, -1.0],
                [ 1.0,  1.0,  1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [-1.0,  1.0, -1.0],
                [-1.0,  1.0,  1.0],
            ],
        ]
    };
}

//-------------------------------------------------------------------

mod tests {
    use super::*;
    use ::algo::group::generate_finite_group;

    use ::std::fmt::Debug;
    use ::std::hash::Hash;
    use ::std::collections::HashSet;

    fn random_x() -> X
    {
        use ::rand::Rng;
        let mut verts = INIT_X.clone();
        ::rand::thread_rng().shuffle(&mut verts);
        verts
    }

    fn validate_homomorphism<G, H>(gdata: DataRef<G>, hdata: DataRef<H>)
    where
        H: Eq + Clone + Debug,
        G: Eq + Clone + Debug + Hash,
    {
        use ::std::collections::HashMap;

        let gs = gdata.members.clone();
        let hs = hdata.members.clone();
        assert_eq!(gs.len(), hs.len());
        let h_by_g: HashMap<_,_> =
            izip!(gs.clone(), hs.clone()).collect();

        for a in 0..gs.len() {
            for b in 0..gs.len() {
                let g = gdata.compose(&gs[a], &gs[b]);
                let h = hdata.compose(&hs[a], &hs[b]);
                assert_eq!(h_by_g[&g], h, "\n{:?}", g);
            }
        }
    }

    type DataRef<G: 'static> = &'static Data<G>;
    struct Data<G: 'static> {
        generators: &'static [G],
        compose: fn(&G, &G) -> G,
        action: fn(&X, &G) -> X,
        members: Vec<G>,
        eye: G,
    }
    impl<G: 'static> Data<G> {
        fn compose(&self, a: &G, b: &G) -> G
        { (self.compose)(a, b) }

        fn action(&self, x: &X, g: &G) -> X
        { (self.action)(x, g) }
    }

    lazy_static! {
        static ref SIGNED_PERM_DATA: Data<SignedPerm> = {
            type G = SignedPerm;
            let eye = SignedPerm::eye();
            let generators = &*SIGNED_PERM_GENERATORS;

            fn compose(a: &G, b: &G) -> G { a.then(b) }
            fn action(x: &X, g: &G) -> X { g.transform(x) }
            let members = generate_finite_group(generators, compose);
            Data { generators, members, compose, action, eye }
        };

        static ref ROTATION_DATA: Data<FracRot> = {
            type G = FracRot;
            let eye = FracRot::eye();
            let generators = &*ROTATION_GENERATORS;

            fn compose(a: &G, b: &G) -> G { a.then(b) }
            fn action(x: &X, g: &G) -> X { g.transform_prim(x) }
            let members = generate_finite_group(generators, compose);
            Data { generators, members, compose, action, eye }
        };

        static ref VERTEX_PERM_DATA: Data<Perm> = {
            type G = Perm;
            let eye = Perm::eye(8);
            let generators = &*VERTEX_PERM_GENERATORS;

            fn compose(a: &G, b: &G) -> G { a.then(b) }
            fn action(x: &X, g: &G) -> X { x.clone().permuted_by(g) }
            let members = generate_finite_group(generators, compose);
            Data { generators, members, compose, action, eye }
        };

        // opposite group;  (a * b) --> (b * a)
        static ref VERTEX_PERM_OP_DATA: Data<Perm> = {
            type G = Perm;
            let eye = VERTEX_PERM_DATA.eye.clone();
            let generators = VERTEX_PERM_DATA.generators;

            fn compose(a: &G, b: &G) -> G { VERTEX_PERM_DATA.compose(b, a) }
            fn action(_: &X, _: &G) -> X { panic!("no action!"); }
            let members = generate_finite_group(generators, compose);
            Data { generators, members, compose, action, eye }
        };
    }

    //---------------------------------------------
    // group properties

    fn test_group_identity<G: PartialEq + Debug>(data: DataRef<G>)
    {
        for g in &data.members {
            assert_eq!(g, &data.compose(&data.eye, g));
            assert_eq!(g, &data.compose(g, &data.eye));
        }
    }

    fn test_associativity<G: Clone + PartialEq + Debug>(data: DataRef<G>)
    {
        for a in &data.members[..] {
            for b in &data.members[..] {
                for c in &data.members[..] {
                    assert_eq!(
                        data.compose(a, &data.compose(b, c)),
                        data.compose(&data.compose(a, b), c));
                }
            }
        }
    }

    fn test_inverse<G: Clone + Debug + Hash + Eq>(data: DataRef<G>)
    {
        let f = data.compose;
        let gs = &data.members[..];
        for a in gs {
            // test latin square principle
            assert_eq!(
                gs.len(),
                gs.iter().map(|b| f(a, b)).collect::<HashSet<_>>().len());
            assert_eq!(
                gs.len(),
                gs.iter().map(|b| f(b, a)).collect::<HashSet<_>>().len());

            // test that left-inverse equals right-inverse.
            let right_inv = gs.iter().map(|b| (b, f(a, b)))
                .find(|&(_, ref c)| c == &data.eye)
                .map(|(b, _)| b).unwrap();

            assert_eq!(f(&right_inv, a), data.eye);
        }
    }

    fn test_group<G: Clone + Debug + Hash + Eq>(data: DataRef<G>)
    {
        test_group_identity(data);
        test_associativity(data);
        test_inverse(data);
    }

    #[test] fn r_is_group()
    { test_group(&ROTATION_DATA) }

    #[test] fn s_is_group()
    { test_group(&SIGNED_PERM_DATA) }

    #[test] fn v_is_group()
    { test_group(&VERTEX_PERM_DATA) }

    #[test] fn vop_is_group()
    { test_group(&VERTEX_PERM_OP_DATA) }

    //---------------------------------------------
    // validate group action properties

    // signed perms, vertex perms, and rotations all admit a group action
    //  on the set of Nx3 matrices vectors consisting of permutations
    //  of the points in [-1, 1]^3

    fn test_action_identity<G: PartialEq + Debug>(data: DataRef<G>)
    {
        let x = random_x();
        assert_eq!(&x, &data.action(&x, &data.eye));
    }

    fn test_compatibility<G: Clone>(data: DataRef<G>)
    {
        let x = random_x();
        for a in &data.members[..] {
            for b in &data.members[..] {
                assert_eq!(
                    data.action(&x, &data.compose(a, b)),
                    data.action(&data.action(&x, a), b));
            }
        }
    }

    fn test_group_action<G: Clone + PartialEq + Debug>(data: DataRef<G>)
    {
        test_action_identity(data);
        test_compatibility(data);
    }

    #[test] fn r_has_action()
    { test_group_action(&ROTATION_DATA) }

    #[test] fn s_has_action()
    { test_group_action(&SIGNED_PERM_DATA) }

    #[test] fn v_has_action()
    { test_group_action(&VERTEX_PERM_DATA) }

    // (vop does not have an action; it fails compatibility)

    //---------------------------------------------
    // basic test of action implementations

    fn test_generator_actions<G: Clone + Debug>(data: DataRef<G>)
    {
        for (g, x) in izip!(data.generators, &*EXPECTED_ACTIONS) {
            println!("{:?}", g);
            assert_eq!(x, &data.action(&INIT_X, g));
        }
    }

    #[test]
    fn action_implementations()
    {
        test_generator_actions(&VERTEX_PERM_DATA);
        test_generator_actions(&SIGNED_PERM_DATA);
        test_generator_actions(&ROTATION_DATA);
        // (vop does not have an action; it fails compatibility)
    }

    //---------------------------------------------
    // equivalence of actions

    #[test]
    fn generated_groups_same_action()
    {
        // NOTE: Vertex perms are not included here because they have
        //       a *different* group action than the others.
        let rots = ROTATION_DATA.members.clone();
        let s_perms = SIGNED_PERM_DATA.members.clone();

        for (s_perm, rot) in izip!(&s_perms, &rots) {
            let s_act = SIGNED_PERM_DATA.action(&INIT_X, s_perm);
            let r_act = ROTATION_DATA.action(&INIT_X, rot);
            assert!(s_act == r_act, "\ns: {:?}\nr: {:?}", s_act, r_act);
        }
    }

    //---------------------------------------------
    // morphisms between groups

    #[test]
    fn conversions_between_generators()
    {
        for i in 0..N_GENERATORS {
            let s_perm = SIGNED_PERM_DATA.generators[i].clone();
            let vop_perm = VERTEX_PERM_OP_DATA.generators[i].clone();
            let rot = ROTATION_DATA.generators[i].clone();

            assert_eq!(s_perm, SignedPerm::from_rot(&rot), "\n{:?}", rot);
            assert_eq!(rot, s_perm.to_rot(), "\n{:?}", s_perm);
            assert_eq!(vop_perm, vertex_perm_from_rot(&rot), "\n{:?}", rot);
        }
    }

    #[test]
    fn conversions_between_generated_groups()
    {
        let rots = ROTATION_DATA.members.clone();
        let s_perms = SIGNED_PERM_DATA.members.clone();
        let vops = VERTEX_PERM_OP_DATA.members.clone();
        assert_eq!(s_perms.len(), 48);
        assert_eq!(vops.len(), 48);
        assert_eq!(rots.len(), 48);

        for (s_perm, vop, rot) in izip!(&s_perms, &vops, &rots) {
            assert_eq!(s_perm, &SignedPerm::from_rot(rot), "\n{:?}", rot);
            assert_eq!(rot, &s_perm.to_rot(), "\n{:?}", s_perm);
            assert_eq!(vop, &vertex_perm_from_rot(rot), "\n{:?}", rot);
        }
    }

    #[test] fn isomorphism_s_r()
    {
        validate_homomorphism(&SIGNED_PERM_DATA, &ROTATION_DATA);
        validate_homomorphism(&ROTATION_DATA, &SIGNED_PERM_DATA);
    }

    #[test] fn isomorphism_s_vop()
    {
        validate_homomorphism(&SIGNED_PERM_DATA, &VERTEX_PERM_OP_DATA);
        validate_homomorphism(&VERTEX_PERM_OP_DATA, &SIGNED_PERM_DATA);
    }

    #[test] fn isomorphism_r_vop()
    {
        validate_homomorphism(&ROTATION_DATA, &VERTEX_PERM_OP_DATA);
        validate_homomorphism(&VERTEX_PERM_OP_DATA, &ROTATION_DATA);
    }

    //---------------------------------------------
    // the final boss: use GroupTree!

    #[test]
    fn group_tree_r_to_vop()
    {
        use ::algo::group::GroupTree;

        // Elide most calls to `vertex_perm_from_rot` using the
        // homomorphism into the opposite group of permutations
        let perms = GroupTree::from_all_members(
            ROTATION_DATA.members.clone(),
            ROTATION_DATA.compose,
        ).compute_homomorphism(
            |g| vertex_perm_from_rot(g),
            |a, b| VERTEX_PERM_OP_DATA.compose(a, b),
        );

        // Do they correctly describe the rotations?
        for (rot, perm) in izip!(&ROTATION_DATA.members, &perms) {
            assert_eq!(
                rot.transform_prim(&INIT_X),
                INIT_X.clone().permuted_by(&perm));
        }
    }
}