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

// FIXME: Taken from an earlier project; not yet pruned for irrelevant stuff.

extern crate num_integer;
extern crate num_traits;
use num_integer::Integer;
use num_traits::{PrimInt,Signed,One};

use ::std::collections::HashMap;
use ::std::hash::Hash;

#[cfg(test)] extern crate rand;

#[derive(Copy,Clone,Debug,Eq,PartialEq)]
pub struct GcdData<X> {
    // greatest common divisor
    pub gcd: X,
    // least common multiple
    pub lcm: X,
    // bezout coefficients
    pub coeffs: (X, X),
    // quotients of the inputs by the GCD
    pub quotients: (X, X),
}

// NOTE:
//  The Signed bound is unavoidable for Extended GCD because the Bezout
//  coefficients can be negative. This is unfortunate for plain old gcd(),
//  which technically shouldn't require the Signed bound.

// Since the bezout coefficients have no impact on each other or on the gcd,
// a sufficiently smart compiler can rip their computations out entirely.
// And as luck would have it, rustc is sufficiently smart!
#[allow(non_snake_case)]
#[inline(always)]
fn extended_gcd__inline<X>(a: X, b: X) -> GcdData<X>
where
    X: PrimInt + Integer + Signed,
{
    let (a_sign, a) = (a.signum(), a.abs());
    let (b_sign, b) = (b.signum(), b.abs());

    // Silly trick because rust doesn't have true multiple assignment:
    // Store the two targets in one variable! Order is (old, current).
    let mut s = (X::one(),  X::zero()); // a coefficient
    let mut t = (X::zero(), X::one());  // b coefficient
    let mut r = (a, b); // gcd
    while r.1 != X::zero() {
        let (div, rem) = (r.0/r.1, r.0%r.1);
        r = (r.1, rem);
        s = (s.1, s.0 - div * s.1);
        t = (t.1, t.0 - div * t.1);
    }

    let quots = (a_sign * t.1.abs(), b_sign * s.1.abs());

    GcdData {
        gcd: r.0,
        // FIXME think more about sign of LCM
        // (current implementation preserves the property a*b == gcd*lcm
        //  which is nice, but I don't know if it is The Right Thing)
        lcm: r.0*quots.0*quots.1,
        coeffs: (a_sign*s.0, b_sign*t.0),
        quotients: quots,
    }
}

/// Compute a greatest common divisor with other miscellany.
pub fn extended_gcd<X>(a: X, b: X) -> GcdData<X>
where
    X: PrimInt + Integer + Signed,
{ extended_gcd__inline(a, b) }

/// Compute a greatest common divisor.
pub fn gcd<X>(a: X, b: X) -> X
where
    X: PrimInt + Integer + Signed,
{ extended_gcd__inline(a, b).gcd }

/// Compute a least common multiple.
pub fn lcm<X>(a: X, b: X) -> X
where
    X: PrimInt + Integer + Signed,
{ extended_gcd__inline(a, b).lcm }

/// Compute a modular multiplicative inverse, if it exists.
///
/// This implementation uses the extended Gcd algorithm,
pub fn inverse_mod<X>(a: X, m: X) -> Option<X>
where
    X: PrimInt + Integer + Signed,
{
    let GcdData { gcd: g, coeffs: (inv, _), .. } = extended_gcd__inline(a, m);
    if g == X::one() { Some(inv.mod_floor(&m)) } else { None }
}

/// Merge many equations of the form `x = ai (mod ni)` into one.
///
/// The moduli don't need to be coprime;
/// ``None`` is returned if the equations are inconsistent.
///
/// `chinese_remainder(vec![])` is defined to be `Some((0,1))`.
pub fn chinese_remainder<X,I>(congruences: I) -> Option<(X,X)>
where
    X: PrimInt + Integer + Signed,
    I: IntoIterator<Item=(X,X)>,
{
    // something something "monadic" something "fold"
    congruences.into_iter().fold(Some((X::zero(),X::one())),
                                 |opt, new_pair| opt.and_then(|acc_pair|
                                     chinese_remainder2(acc_pair, new_pair)
                                 )
    )
}

/// Merge two equations of the form ``x = ai (mod ni)`` into one.
///
/// The moduli don't need to be coprime;
/// `None` is returned if the equations are inconsistent.
///
/// Panics if a modulus is negative or zero.
pub fn chinese_remainder2<X>((a1,n1):(X,X), (a2,n2):(X,X)) -> Option<(X,X)>
where
    X: PrimInt + Integer + Signed,
{
    // I'm too lazy right now to consider whether there is a
    //   reasonable behavior for negative moduli
    assert!(n1.is_positive());
    assert!(n2.is_positive());

    let GcdData {
        gcd: g,
        lcm: n3,
        coeffs: (c1,c2),
        ..
    } = extended_gcd__inline(n1, n2);

    let (a1div, a1rem) = a1.div_rem(&g);
    let (a2div, a2rem) = a2.div_rem(&g);
    if a1rem != a2rem { None }
        else {
            let a3 = (a2div*c1*n1 + a1div*c2*n2 + a1rem).mod_floor(&n3);
            Some((a3, n3))
        }
}

// used for conversions of literals (which will clearly never fail)
fn lit<X>(x: i64) -> X where X: PrimInt { X::from(x).unwrap() }
// slightly more verbose for use outside mathematical expressions
fn convert<X>(x: i64) -> X where X: PrimInt { X::from(x).unwrap() }


/// An argument to tame function count explosion for functions
/// which can optionally deal with permutation symmetry.
#[derive(Copy,Clone,Hash,PartialEq,Eq,Debug)]
pub enum OrderType {
    /// Order matters; consider all distinct permutations.
    Ordered,
    /// Order does not matter; only consider distinct combinations.
    Unordered,
}

/// Used as a half-baked alternative to writing a generic interface
/// over RangeTo and RangeToInclusive
#[derive(Copy,Clone,Hash,PartialEq,Eq,Debug)]
pub enum UpperBound<X> { Upto(X), Under(X), }
impl<X> UpperBound<X> where X: Integer + One,
{
    // FIXME: I think it is hard to read code that uses this.
    /// a generic form of  (min..x).next_back() or (min...x).next_back()
    fn inclusive_limit_from(self, min: X) -> Option<X> {
        match self {
            Under(upper) => if min >= upper { None } else { Some(upper - One::one()) },
            Upto(max) =>    if min > max { None } else { Some(max) },
        }
    }
}

#[test]
fn test_inclusive_limit_from() {
    assert_eq!(Upto(4).inclusive_limit_from(3), Some(4));
    assert_eq!(Upto(3).inclusive_limit_from(3), Some(3));
    assert_eq!(Upto(2).inclusive_limit_from(3), None);
    assert_eq!(Upto(1).inclusive_limit_from(3), None);

    assert_eq!(Under(5).inclusive_limit_from(3), Some(4));
    assert_eq!(Under(4).inclusive_limit_from(3), Some(3));
    assert_eq!(Under(3).inclusive_limit_from(3), None);
    assert_eq!(Under(2).inclusive_limit_from(3), None);

    // no underflow please kthx
    assert_eq!(Under(0).inclusive_limit_from(0), None);
}

use UpperBound::*;
use OrderType::*;

// NOTE: Further possible generalizations:
//  * Take a Range instead of UpperBound so that zero can optionally be included
//    (however it would also require solving how to produce correct results
//     for all other lower bounds)
//  * count_coprime_tuplets(max, n)
/// Counts coprime pairs of integers `>= 1`
///
/// # Notes
///
/// `(0,1)` and `(1,0)` are not counted.
/// `(1,1)` (the only symmetric pair) is counted once.
///
/// # Reference
///
/// http://laurentmazare.github.io/2014/09/14/counting-coprime-pairs/
pub fn count_coprime_pairs<X>(bound: UpperBound<X>, order_type: OrderType) -> X
where
    X: PrimInt + Integer + Hash,
    X: ::std::fmt::Debug,
{
    let max = {
        if let Some(max) = bound.inclusive_limit_from(lit(1)) { max }
            else { return lit(0); } // catches Under(0), Upto(0), Under(1)
    };

    let ordered = count_ordered_coprime_pairs(max);
    match order_type {
        Ordered => ordered,
        // Every combination was counted twice except for (1,1);
        // Add 1 so that they are ALL double counted, then halve it.
        // (this also fortituously produces 0 for max == 0)
        Unordered => (ordered + lit(1)) / lit(2),
    }
}

fn count_ordered_coprime_pairs<X>(max: X) -> X
where
    X: PrimInt + Integer + Hash,
    X: ::std::fmt::Debug,
{
    // Function can be described with this recursion relation:
    //
    //     c(n) = n**2 - sum_{k=2}^n c(n // k)
    //
    // where '//' is floored division.
    //
    // Many values of k share the same value of (n // k),
    //  thereby permitting a coarse-graining approach.

    // unique values of m (=floor(n/k)) for small k
    let fine_deps = |n| {
        (2..).map(convert::<X>)
            .map(move |k| n/k).take_while(move |&m| m*m > n)
    };

    // values of m (=floor(n/k)) shared by many large k.
    let coarse_deps = |n| {
        (1..).map(convert::<X>)
            .take_while(move |&m| m*m <= n)
            // don't produce m == 1 for n == 1
            .skip_while(move |_| n == lit(1))
    };

    let coarse_multiplicity = |n,m| n/m - n/(m + lit(1));

    // Get all values that need to be computed at some point.
    //
    // Interestingly, these are just 'max' and its direct dependencies
    // (which are of the form 'max // k').  The reason the subproblems
    // never introduce new dependencies is because integer division
    // apparently satisfies the following property for x non-negative
    // and a,b positive:
    //
    //     (x // a) // b  ==  (x // b) // a  ==  x // (a*b)
    //
    // (NOTE: euclidean division wins *yet again*; it is the only sign convention
    //  under which this also works for negative 'x', 'a', and 'b'!)
    let order = {
        let mut vec = vec![max];
        vec.extend(fine_deps(max));
        vec.extend(coarse_deps(max));
        vec.sort();
        vec
    };

    let mut memo = HashMap::new();
    let compute = |n, memo: &HashMap<X,X>| {
        let acc = n*n;

        let acc = {
            coarse_deps(n)
                .map(|m| memo[&m.into()] * coarse_multiplicity(n,m))
                .fold(acc, |a,b| a-b)
        };

        let acc = {
            fine_deps(n)
                .map(|m| memo[&m.into()])
                .fold(acc, |a,b| a-b)
        };

        acc
    };

    for x in order {
        let value = compute(x, &memo);
        memo.insert(x, value);
    }
    memo[&max]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::num_integer::Integer;
    use ::num_traits::{PrimInt,Signed};

    use test;
    use rand::{Rng};

    #[test]
    fn test_gcd() {
        // swap left/right
        // (using a pair that takes several iterations)
        assert_eq!(gcd(234,123), 3);
        assert_eq!(gcd(123,234), 3);

        // negatives
        assert_eq!(gcd(-15,20), 5);
        assert_eq!(gcd(15,-20), 5);
        assert_eq!(gcd(-15,-20), 5);

        // zeroes
        assert_eq!(gcd(0,17), 17);
        assert_eq!(gcd(17,0), 17);
        assert_eq!(gcd(0,0), 0);
    }

    #[test]
    fn test_chinese_remainder() {
        // test both interfaces
        let eq1 = (2328,16256);
        let eq2 = (410,5418);
        let soln = (28450328, 44037504);
        assert_eq!(chinese_remainder2(eq1,eq2), Some(soln));
        assert_eq!(chinese_remainder(vec![eq1,eq2]), Some(soln));

        // (0,1) serves as an "identity"
        assert_eq!(chinese_remainder(vec![]), Some((0,1)));
        assert_eq!(chinese_remainder(vec![(0,1)]), Some((0,1)));
        assert_eq!(chinese_remainder(vec![(0,1),(13,36)]), Some((13,36)));
        assert_eq!(chinese_remainder(vec![(13,36),(0,1)]), Some((13,36)));

        // single equation
        assert_eq!(chinese_remainder(vec![eq1]), Some(eq1));

        // inconsistent equations
        assert_eq!(chinese_remainder2((10,7),(4,14)), None);
        assert_eq!(chinese_remainder(vec![(10,7),(4,14)]), None);

        // FIXME: test more than 2 equations
        // FIXME: do we specify behavior for when the input a_i are
        //   not already reduced modulo n_i?
    }

    #[test]
    fn test_inverse_mod() {
        let solns15 = vec![
            None, Some(1),  Some(8),  None,    Some(4),
            None, None,     Some(13), Some(2), None,
            None, Some(11), None,     Some(7), Some(14),
        ];
        for x in -15..30 {
            assert_eq!(inverse_mod(x,15), solns15[x.mod_floor(&15) as usize]);
        }
    }

    #[test]
    fn test_count_coprime_pairs() {
        fn check<X>(bound: UpperBound<X>, expect_o: X, expect_u: X)
            where X: ::std::fmt::Debug + PrimInt + ::std::hash::Hash + Integer {
            let actual_o = count_coprime_pairs(bound, Ordered);
            let actual_u = count_coprime_pairs(bound, Unordered);
            assert_eq!(
                actual_o, expect_o,
                "g({:?}, Ordered) == {:?}, != {:?}",
                bound, actual_o, expect_o,
            );
            assert_eq!(
                actual_u, expect_u,
                "g({:?}, Unordered) == {:?}, != {:?}",
                bound, actual_u, expect_u,
            );
        };

        // special-ish cases
        check(Under(0u32), 0, 0); // unsigned to check for underflow
        check(Under(0i32), 0, 0); // signed to check for poor usage of checked_sub
        check(Upto(0u32), 0, 0);
        check(Upto(0i32), 0, 0);
        check(Upto(1u32), 1, 1);
        check(Upto(1i32), 1, 1);
        // a nontrivial coprime pair (2,3)
        check(Upto(3u32), 7, 4);
        // a nontrivial non-coprime pair (2,4)
        check(Upto(4u32), 11, 6);
        // problem size large enough to test both fine-graining and coarse-graining
        check(Upto(100u32), 6087, 3044);
        // a biggun
        assert_eq!(count_coprime_pairs(Upto(10_000_000i64), Ordered), 60792712854483i64);

        // try a variety of bounds in an attempt to get memo[&x] to panic
        //  on a missed dependency
        let mut rng = ::rand::thread_rng();
        println!("{}", // prevent optimizing out
            (0..100).map(|_| rng.gen_range(100, 100_000i64))
                .map(|x| count_coprime_pairs(Upto(x), Ordered))
                .sum::<i64>(),
        );
    }

    // Gold standard for binary comparison.
    #[inline(never)]
    fn gcd__reference<X>(a: X, b: X) -> X
    where
        X: PrimInt + Integer + Signed,
    {
        let mut a = a.abs();
        let mut b = b.abs();
        while b != X::zero() {
            let tmp = b;
            b = a % b;
            a = tmp;
        }
        a
    }

    // Impressively, rustc grinds this down to a *byte-perfect match*
    //  against gcd__reference.
    #[inline(never)]
    fn gcd__optimized<X>(a: X, b: X) -> X
    where
        X: PrimInt + Integer + Signed,
    { gcd(a, b) }

    // force the two inline(never) functions above to be compiled
    #[test]
    fn dummy__compile_testfuncs() {
        assert_eq!(gcd__reference(15,20), 5);
        assert_eq!(gcd__optimized(20,15), 5);
        // Interestingly, the compiled inline(never) functions will
        //  recieve optimizations based on their inputs.
        // Without these following invocations, rustc will compile
        //  faster versions that only support positive arguments.
        assert_eq!(gcd__reference(-15,-20), 5);
        assert_eq!(gcd__optimized(-20,-15), 5);
    }
}
