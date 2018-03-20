
//!
//! Citations:
//!
//! * B. Gruber, "The Relationship between Reduced Cells
//!   in a General Bravais lattice." Acta Crystallographica
//!   Section A 29 (1973): 433-440.
//!
//! * Grosse-Kunstleve, Ralf W., Nicholas K. Sauter,
//!   and Paul D. Adams. "Numerically stable algorithms
//!   for the computation of reduced unit cells."
//!   Acta Crystallographica Section A: Foundations of
//!   Crystallography 60.1 (2004): 1-6.

use ::{Lattice};

use ::rsp2_array_utils::{arr_from_fn};
use ::rsp2_array_types::{V3, M33, Envee, Unvee, dot, mat, inv};

use ::std::cmp::Ordering;

// TODO:
// * Finish writing tests like those in Kuntsleve
// * gen all possible rotations of niggli cells in frac space
//   (unimodular matrices of elements {-2, -1, 0, 1, 2}.
//    note that they are NOT necessarily orthogonal, but rather
//    they are such that ``L C^T C L^T == L L^T``)

#[derive(Debug, Copy, Clone)]
struct Fuzz {
    tol: f64,
}

impl Fuzz {
    pub fn from_volume(vol: f64) -> Fuzz
    { Fuzz { tol: 1e-5 * vol.abs().cbrt() } }

    pub fn lt(&self, x: f64, y: f64) -> bool
    { x < y - self.tol }
    pub fn gt(&self, x: f64, y: f64) -> bool
    { self.lt(y, x) }

    pub fn le(&self, x: f64, y: f64) -> bool
    { ! self.gt(x, y) }
    pub fn ge(&self, x: f64, y: f64) -> bool
    { ! self.lt(x, y) }

    pub fn eq(&self, x: f64, y: f64) -> bool
    { ! self.lt(x, y) && ! self.gt(x, y) }

    pub fn cmp(&self, x: f64, y: f64) -> Ordering
    {
        if self.lt(x, y) { Ordering::Less }
        else if self.gt(x, y) { Ordering::Greater }
        else { Ordering::Equal }
    }
}

pub use self::unimodular::Unimodular;
use self::unimodular::UnimodularState;
mod unimodular {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Unimodular {
        matrix: M33<i32>,
        inverse: M33<i32>,
    }

    impl Unimodular {
        #[inline] pub fn matrix(&self) -> &M33<i32> { &self.matrix }
        #[inline] pub fn inverse_matrix(&self) -> &M33<i32> { &self.inverse }
    }

    // easier to update
    #[derive(Debug, Clone)]
    pub(super) struct UnimodularState(pub(super) M33<i32>);

    impl UnimodularState {
        pub fn eye() -> Self
        { UnimodularState(mat::from_array([[1,0,0], [0,1,0], [0,0,1]])) }

        // steps N1, N2
        /// Swap two rows.
        #[inline]
        pub fn row_swap(&mut self, j: usize, k: usize)
        {
            // NOTE: Can't mem::swap because of simultaneous indexing
            let ghost = self.0;
            self.0[j] = ghost[k];
            self.0[k] = ghost[j];
        }

        // steps B2-B5
        /// Add a multiple of one lattice vector to a different one.
        #[inline]
        pub fn row_axpy(&mut self, to: usize, mul: i32, from: usize)
        {
            assert_ne!(from, to, "adding a row to itself is not a unimodular operation");
            self.0[to] += mul * self.0[from];
        }

        // steps N3
        /// Negate a lattice vector.
        #[inline]
        pub fn row_negate(&mut self, row: usize)
        { self.0[row] *= -1; }

        pub fn finish(&self) -> Unimodular
        {
            // FIXME it feels cleaner to compute the inverse alongside
            //       the matrix rather than to do a float inversion at the end
            let floats_inv = inv(&self.0.map(|x| x as f64));
            let inverse = ::util::Tol(1e-6).unfloat_m33(&floats_inv).expect("bug!");

            Unimodular { matrix: self.0, inverse }
        }
    }
}

// a small inner module to let privacy assist in
//  protecting some invariants
use self::state::State;
mod state {
    use super::*;

    #[derive(Debug, Clone)]
    pub(super) struct State {
        // FIXME mixture of different types of state

        // constant state
        original: Lattice,
        fuzz: Fuzz,

        // mutatable state
        unimodular: UnimodularState,

        // precomputed data
        // (invariant: these are always updated alongside unimodular)
        lattice: M33,
        abc: [f64; 3],
        xyz: [f64; 3],
    }

    impl State {
        pub fn new(lattice: &Lattice) -> Self
        { Self::from_matrices(lattice, &UnimodularState::eye(), Fuzz::from_volume(lattice.volume())) }

        fn from_matrices(original: &Lattice, unimodular: &UnimodularState, fuzz: Fuzz) -> Self
        {
            let original = original.clone();
            let unimodular = unimodular.clone();

            let unimodular_float = unimodular.0.map(Into::into);
            let lattice = &unimodular_float * original.matrix();

            let abc = arr_from_fn(|k| dot(&lattice[k], &lattice[k]));
            let xyz = arr_from_fn(|k| 2.0 * dot(&lattice[(k + 1) % 3], &lattice[(k + 2) % 3]));
            State { original, unimodular, lattice, abc, xyz, fuzz }
        }

        pub fn lattice_matrix(&self) -> &M33 { &self.lattice }
        pub fn unimodular_matrix(&self) -> &M33<i32> { &self.unimodular.0 }
        pub fn fuzz(&self) -> Fuzz { self.fuzz }
        pub fn abc(&self) -> &[f64; 3] { &self.abc }
        pub fn xyz(&self) -> &[f64; 3] { &self.xyz }
        pub fn a(&self) -> f64 { self.abc()[0] }
        pub fn b(&self) -> f64 { self.abc()[1] }
        pub fn c(&self) -> f64 { self.abc()[2] }
        pub fn x(&self) -> f64 { self.xyz()[0] }
        pub fn y(&self) -> f64 { self.xyz()[1] }
        pub fn z(&self) -> f64 { self.xyz()[2] }

        pub fn change_basis<F>(&mut self, f: F)
        where F: FnOnce(&mut UnimodularState)
        {
            // change basis
            f(&mut self.unimodular);

            // update precomputed data
            *self = Self::from_matrices(&self.original, &self.unimodular, self.fuzz);
        }

        pub fn finish(self) -> LatticeReduction
        { LatticeReduction {
            original: self.original,
            transform: self.unimodular.finish(),
            reduced: Lattice::new(&self.lattice),
        }}
    }
}


// Algorithm N of B. Gruber (1973),
// with interpretations from R. W. Grosse-Kunstleve (2004)
fn normalize_characteristic(state: &mut State)
{
    let fuzz = state.fuzz();

    //------------
    // Steps N1-N2.
    // These look like some kind of bubble sort.

    loop {
        let mut maybe_swap = |j: usize, k: usize| {
            let abc = state.abc().clone();
            let xyz = state.xyz().clone();

            // (note: this matches the Algol 60 in the footnote on page 433.
            //        Like modern languages, Algol gave higher precedence to AND.)
            let do_it = false
                || fuzz.gt(abc[j], abc[k])
                || fuzz.eq(abc[j], abc[k]) && fuzz.gt(xyz[j].abs(), xyz[k].abs());

            if do_it {
                state.change_basis(|u| u.row_swap(j, k));
            }
            do_it
        };

        // Step N1
        maybe_swap(0, 1);

        // Step N2
        if maybe_swap(1, 2) { continue; }
        else { break; }
    }

    //------------
    // Step N3
    // Make all offdiagonals the same sign (if they are all nonzero). (?)

    // This is cribbed from the 'cctbx/uctbx' python code
    //  (which is associated with Grosse-Kunstleve (2004))

    let xyz = state.xyz().clone();
    match fuzz.cmp(xyz[0] * xyz[1] * xyz[2], 0.0) {
        Ordering::Equal => {},

        Ordering::Less => {
            state.change_basis(|u| {
                for k in 0..3 {
                    if fuzz.lt(xyz[k], 0.0) {
                        u.row_negate(k);
                    }
                }
            });
        },

        Ordering::Greater => {

            // Grosse-Kunstleve (2004) appears to contain a typo
            // (using ζ instead of ξ when working
            //   with 'i' in formula (8)).
            //
            // The following reflects the python code
            //   at cctbx/uctbx/reduction_base.py
            state.change_basis(|u| {

                let mut flips = 0;
                let mut flip_me = None;
                for k in 0..3 {
                    match fuzz.cmp(xyz[k], 0.0) {
                        Ordering::Greater => {
                            u.row_negate(k);
                            flips += 1;
                        },
                        Ordering::Equal => {
                            flip_me = Some(k);
                        },
                        Ordering::Less => {},
                    }
                }

                if flips % 2 == 0 {
                    let flip_me = flip_me.expect("bug! (flip_me without zeros?)");
                    u.row_negate(flip_me);
                }
            })
        },
    };
}

/// A pairing of a lattice matrix with its reduced form,
/// along with the integer coefficient matrices that convert
/// between the two.
#[derive(Debug, Clone)]
pub struct LatticeReduction {
    original: Lattice,
    transform: Unimodular,
    reduced: Lattice,
}

impl LatticeReduction {
    #[inline] pub fn original(&self) -> &Lattice { &self.original }
    #[inline] pub fn reduced(&self) -> &Lattice { &self.reduced }
    #[inline] pub fn transform(&self) -> &Unimodular { &self.transform }
    // TODO: expose transform in reasonable manner
}

pub fn algorithm_b(lattice: &Lattice) -> LatticeReduction
{
    let mut state = State::new(lattice);
    let fuzz = state.fuzz();

    'lets_try_that_again:
    loop {
        // B1
        normalize_characteristic(&mut state);

        // B2-B4.
        // I wouldn't bother trying to refactor these.
        let (a, b, _c) = tup3(state.abc());
        let (x, y, z) = tup3(state.xyz());

        // B2
        let do_it = fuzz.gt(x.abs(), b)
            || fuzz.eq(x,  b) && fuzz.gt(z, 2.0 * y)
            || fuzz.eq(x, -b) && fuzz.lt(z, 0.0);

        if do_it {
            state.change_basis(|u| {
                let mul = -((x + b) / (2.0 * b)).floor();
                assert_ne!(mul, 0.0);
                u.row_axpy(2, mul as i32, 1);
            });
            continue 'lets_try_that_again;
        }

        // B3
        let do_it = fuzz.gt(y.abs(), a)
            || fuzz.eq(y,  a) && fuzz.gt(z, 2.0 * x)
            || fuzz.eq(y, -a) && fuzz.lt(z, 0.0);

        if do_it {
            state.change_basis(|u| {
                let mul = -((y + a) / (2.0 * a)).floor();
                assert_ne!(mul, 0.0);
                u.row_axpy(2, mul as i32, 0);
            });
            continue 'lets_try_that_again;
        }

        // B4
        let do_it = fuzz.gt(z.abs(), a)
            || fuzz.eq(z,  a) && fuzz.gt(y, 2.0 * x)
            || fuzz.eq(z, -a) && fuzz.lt(y, 0.0);

        if do_it {
            state.change_basis(|u| {
                let mul = -((z + a) / (2.0 * a)).floor();
                assert_ne!(mul, 0.0);
                u.row_axpy(1, mul as i32, 0);
            });
            continue 'lets_try_that_again;
        }

        // B5
        let xyzab = x + y + z + a + b;
        let aayyz = 2.0 * (a + y) + z;
        let do_it = fuzz.lt(xyzab, 0.0)
            || fuzz.eq(xyzab, 0.0) && fuzz.gt(aayyz, 0.0);

        if do_it {
            state.change_basis(|u| {
                let aayyzz = aayyz + z; // *shrug*
                let mul = -(xyzab / aayyzz).floor();
                assert_ne!(mul, 0.0);
                u.row_axpy(1, mul as i32, 0);
            });
            continue 'lets_try_that_again;
        }

        break;
    }

    // Grosse-Kunstleve (2004) kept determinant positive by negating
    //  some matrices to fix the signs, which works since the dimension
    //  of 3 is odd. We'll just negate the final result instead.
    match state.unimodular_matrix().det() {
        1 => {},
        -1 => {
            state.change_basis(|u| {
                u.row_negate(0);
                u.row_negate(1);
                u.row_negate(2);
            })
        },
        d => panic!("Bad unimodular determinant: {}", d),
    }

    state.finish()
}

pub mod conditions {
    use super::*;

    trait Implies: Sized { fn implies(self, other: bool) -> bool; }
    impl Implies for bool {
        fn implies(self, other: bool) -> bool
        { ! (self && !other) }
    }

    // FIXME intly-typed enum
    fn kind(state: &State) -> u8
    {
        let fuzz = state.fuzz();
        match state.xyz().iter().filter(|&&x| fuzz.ge(x, 0.0)).count() {
            3 => 1,
            0 => 2,
            _ => 0,
        }
    }

    // cctbx/uctbx/reduction_base.py
    fn primary(state: &State) -> bool
    {
        let fuzz = state.fuzz();
        let ((a, b, c), (x, y, z)) = (tup3(state.abc()), tup3(state.xyz()));

        true
        && fuzz.le(a, b)
        && fuzz.le(b, c)
        && fuzz.le(x.abs(), b)
        && fuzz.le(y.abs(), a)
        && fuzz.le(z.abs(), a)
    }

    // cctbx/uctbx/reduction_base.py
    fn main(state: &State) -> bool
    {
        primary(state)
        && match kind(state) {
            0 => false,
            1 => true,
            2 => {
                let xyzab = state.x() + state.y() + state.z() + state.a() + state.b();
                state.fuzz().ge(xyzab, 0.0)
            }
            _ => unreachable!(),
        }
    }

    // cctbx/uctbx/reduction_base.py
    fn buerger(state: &State) -> bool
    {
        let ((a, b, c), (x, y, z)) = (tup3(state.abc()), tup3(state.xyz()));
        let fuzz = state.fuzz();

        main(state)
        && fuzz.eq(a, b).implies(fuzz.le(x.abs(), y.abs()))
        && fuzz.eq(b, c).implies(fuzz.le(y.abs(), z.abs()))
    }

    // cctbx/uctbx/reduction_base.py
    fn niggli(state: &State) -> bool
    {
        let ((a, b, c), (d, e, f)) = (tup3(state.abc()), tup3(state.xyz()));
        let fuzz = state.fuzz();

        buerger(state)
        && fuzz.eq(d, b).implies(fuzz.le(f, 2.0 * e))
        && fuzz.eq(e, a).implies(fuzz.le(f, 2.0 * d))
        && fuzz.eq(f, a).implies(fuzz.le(e, 2.0 * d))
        && fuzz.eq(d, -b).implies(fuzz.eq(f, 0.0))
        && fuzz.eq(e, -a).implies(fuzz.eq(f, 0.0))
        && fuzz.eq(f, -a).implies(fuzz.eq(e, 0.0))
        && fuzz.eq(d + e + f + a + b, 0.0).implies(fuzz.le(2.0 * (a + e) + f, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radians(x: f64) -> f64
    { x * (::std::f64::consts::PI / 180.0) }

    fn lattice_from_params(lengths: &[f64; 3], angles: &[f64; 3]) -> [[f64; 3]; 3]
    {
        // from pymatgen
        let (a, b, c) = tup3(lengths);
        let (alpha, beta, gamma) = tup3(angles);
        let numer = alpha.cos() * beta.cos() - gamma.cos();
        let denom = alpha.sin() * beta.sin();
        // Sometimes rounding errors result in values slightly > 1.
        let gamma_star = (numer / denom).max(-1.0).min(1.0).acos();

        let av = [a * beta.sin(), 0.0, a * beta.cos()];
        let bv = [
            -b * alpha.sin() * gamma_star.cos(),
            b * alpha.sin() * gamma_star.sin(),
            b * alpha.cos(),
        ];
        let cv = [0.0, 0.0, c];
        [av, bv, cv]
    }

    #[cfg(nope)]
    #[test]
    fn gk_2004_test_2()
    {
        let lengths = vec![10.0, 20.0, 30.0];
        let angles: Vec<_> =
            [10.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 170.0]
            .iter().map(|&x| radians(x))
            .collect();

        // with apologies to rustfmt
        for &a in &lengths {
        for &b in &lengths {
        for &c in &lengths {
        for &alpha in &angles {
        for &beta in &angles {
        for &gamma in &angles {
            let lattice = lattice_from_params(&[a, b, c], &[alpha, beta, gamma]);
            unimplemented!();
        }}}}}}
    }
}


//--------------------

fn tup3<T:Copy>(x: &[T; 3]) -> (T, T, T)
{ (x[0], x[1], x[2]) }
