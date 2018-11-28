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

use crate::Idx;
use std::marker::PhantomData;
use std::mem;

enum Void {}
pub struct KeepIt(Void);
pub struct CastIt<V: ?Sized>(PhantomData<V>, Void);

/// Trait for `index_cast`. See the free function for details.
///
/// # Safety
///
/// This is unsafe because it has generic impls that perform transmute.
///
/// Special consideration must be given to types whose data layout is affected
/// by trait impls, such as HashMap and BTreeMap.
///
/// The following conditions must be met for an impl to be safe:
///
/// * `Self` and `Result` must have identical representations.
/// * If `Self` implements any of the traits `PartialEq`, `Eq`, `Hash`,
///   `PartialOrd`, and `Ord`, then so must `Result` with an identical implementation
///   (and vice versa).
///     * This implies that generic impls of these traits on implementors of IndexCast
///       must not be marked `default`.
///
/// Notice that this trait is not limited to types for which a safe conversion
/// from `Self` to `Result` exists, as it can, for instance, convert `&A` to `&B`.
pub unsafe trait IndexCast<Result: ?Sized, Disambig: ?Sized> {
    fn index_cast(self) -> Result
    where Self: Sized, Result: Sized,
    { unsafe { mem::transmute_copy(&mem::ManuallyDrop::new(self)) } }
}

/// Takes a single index type embedded somewhere in a larger type, and
/// casts it into another index type. (e.g. `&[Vec<(usize, f64)>]`
/// to `&[Vec<(MyIdx, f64)>]`).
///
/// This uses overlapping impls with an extra type parameter to serve
/// as a disambiguator (the same technique used to implement HLists in
/// Rust).  As long as you annotate the output type, and there is only
/// one index type that changes from the input to the output, type
/// inference will compute the disambiguator.
pub fn index_cast<V, Result, Disambig: ?Sized>(value: V) -> Result
where V: IndexCast<Result, Disambig>,
{ IndexCast::index_cast(value) }

// !!!!!!!!!!!!!!!
// So... basically there's no way to make this work for a large variety of
// types without writing implementations for each one.
//
// Let's just be pragmatic here.  If you use cast_index on a type and it doesn't
// work, add an impl.  Otherwise, who cares?
// !!!!!!!!!!!!!!!

// base case for a successful cast.
//
// NOTE: No other base cases may exist, for safety purposes.
unsafe impl<Src: Idx, Dest: Idx> IndexCast<Dest, KeepIt> for Src {}

// mods are just for organization
// (and, maybe you have an IDE that can collapse them)

mod basic {
    use super::*;

    unsafe impl<A1, B1, Dis> IndexCast<(B1, ), (Dis, )> for (A1, )
    where A1: IndexCast<B1, Dis> {}

    unsafe impl<A1, A2, B, Dis> IndexCast<(B, A2), (CastIt<Dis>, KeepIt)> for (A1, A2)
    where A1: IndexCast<B, Dis> {}

    unsafe impl<A1, A2, B, Dis> IndexCast<(A1, B), (KeepIt, CastIt<Dis>)> for (A1, A2)
    where A2: IndexCast<B, Dis> {}

    unsafe impl<A1, A2, A3, B, Dis> IndexCast<(B, A2, A3), (CastIt<Dis>, KeepIt, KeepIt)> for (A1, A2, A3)
    where A1: IndexCast<B, Dis> {}

    unsafe impl<A1, A2, A3, B, Dis> IndexCast<(A1, B, A3), (KeepIt, CastIt<Dis>, KeepIt)> for (A1, A2, A3)
    where A2: IndexCast<B, Dis> {}

    unsafe impl<A1, A2, A3, B, Dis> IndexCast<(A1, A2, B), (KeepIt, KeepIt, CastIt<Dis>)> for (A1, A2, A3)
    where A3: IndexCast<B, Dis> {}

    unsafe impl<'a, A: ? Sized, B: ? Sized, Dis: ? Sized> IndexCast<&'a B, &'a Dis> for &'a A
    where A: IndexCast<B, Dis> {}

    unsafe impl<'a, A: ? Sized, B: ? Sized, Dis: ? Sized> IndexCast<&'a mut B, &'a mut Dis> for &'a mut A
    where A: IndexCast<B, Dis> {}

    unsafe impl<A, B, Dis> IndexCast<[B], [Dis]> for [A]
    where A: IndexCast<B, Dis> {}

    unsafe impl<A, B, Dis> IndexCast<Vec<B>, Vec<Dis>> for Vec<A>
    where A: IndexCast<B, Dis> {}
}

mod collections {
    use super::*;
    use std::collections::{HashMap, BTreeMap};
    use std::collections::{HashSet, BTreeSet};

    // The safety of these relies on:
    // * IndexCast being an unsafe trait that requires Self and Result to have identical
    //   impls for these traits.
    // * Idx being an unsafe trait which requires types to implement PartialEq, Eq,
    //   PartialOrd, Ord, and Hash identically to usize.
    // * The wording of that fact being such that there is no loophole like
    //   converting `A -> B -> C` where `B: !Hash`.

    unsafe impl<A, B, V, Dis> IndexCast<BTreeMap<B, V>, BTreeMap<CastIt<Dis>, KeepIt>> for BTreeMap<A, V>
    where A: IndexCast<B, Dis> {}

    unsafe impl<A, B, K, Dis> IndexCast<BTreeMap<K, B>, BTreeMap<KeepIt, CastIt<Dis>>> for BTreeMap<K, A>
    where A: IndexCast<B, Dis> {}

    unsafe impl<A, B, V, Dis> IndexCast<HashMap<B, V>, HashMap<CastIt<Dis>, KeepIt>> for HashMap<A, V>
    where A: IndexCast<B, Dis> {}

    unsafe impl<A, B, K, Dis> IndexCast<HashMap<K, B>, HashMap<KeepIt, CastIt<Dis>>> for HashMap<K, A>
    where A: IndexCast<B, Dis> {}

    unsafe impl<A, B, Dis> IndexCast<HashSet<B>, HashSet<Dis>> for HashSet<A>
    where A: IndexCast<B, Dis> {}

    unsafe impl<A, B, Dis> IndexCast<BTreeSet<B>, BTreeSet<Dis>> for BTreeSet<A>
    where A: IndexCast<B, Dis> {}
}

#[test]
fn test_it() {
    newtype_index!(A);
    newtype_index!(B);
    newtype_index!(C);

    let x: &usize = &0;
    let _: &A = index_cast(x);

    // explicit disambiguator.  All instances of KeepIt can be inferred.
    let x: &[(f64, Vec<(B, A, f64)>)] = &[];
    let _: &[(f64, Vec<(B, C, f64)>)] = index_cast::<_, _, &[(_, CastIt<Vec<(_, CastIt<_>, _)>>)]>(x);

    // actually, that disambiguator was not necessary.
    let x: &[(f64, Vec<(B, A, f64)>)] = &[];
    let _: &[(f64, Vec<(B, C, f64)>)] = index_cast(x);
}

#[allow(unused)] // compiletest
fn holy_shnozballs_it_even_works_in_generic_contexts<I: Idx, J: Idx>()
{
    use std::collections::BTreeMap;
    newtype_index!(B);

    let x: &[(f64, Vec<(B, BTreeMap<usize, I>, f64)>)] = &[];
    let _: &[(f64, Vec<(B, BTreeMap<usize, J>, f64)>)] = index_cast(x);
}
