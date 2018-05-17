use ::Idx;
use ::std::marker::PhantomData;
use ::std::mem;

enum Void {}
pub struct KeepIt(Void);
pub struct CastIt<V: ?Sized>(PhantomData<V>, Void);

/// Trait for `cast_index`. See that method for details.
///
/// # Safety
///
/// This is unsafe because it has generic impls that perform transmute.
pub unsafe trait CastIndex<Result: ?Sized, Disambig: ?Sized> {
    fn cast_index(self) -> Result
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
pub fn cast_index<V, Result, Disambig: ?Sized>(value: V) -> Result
where V: CastIndex<Result, Disambig>,
{ CastIndex::cast_index(value) }

// !!!!!!!!!!!!!!!
// So... basically there's no way to make this work for a large variety of
// types without writing implementations for each one.
//
// Let's just be pragmatic here.  If you use cast_index on a type and it doesn't
// work, add an impl.  Otherwise, who cares?
// !!!!!!!!!!!!!!!

// base case for a successful cast
unsafe impl<Src: Idx, Dest: Idx> CastIndex<Dest, KeepIt> for Src {}

// mods are just for organization
// (and, maybe you have an IDE that can collapse them)

mod basic {
    use super::*;

    unsafe impl<A1, B1, Dis> CastIndex<(B1, ), (Dis, )> for (A1, )
    where A1: CastIndex<B1, Dis> {}

    unsafe impl<A1, A2, B, Dis> CastIndex<(B, A2), (CastIt<Dis>, KeepIt)> for (A1, A2)
    where A1: CastIndex<B, Dis> {}

    unsafe impl<A1, A2, B, Dis> CastIndex<(A1, B), (KeepIt, CastIt<Dis>)> for (A1, A2)
    where A2: CastIndex<B, Dis> {}

    unsafe impl<A1, A2, A3, B, Dis> CastIndex<(B, A2, A3), (CastIt<Dis>, KeepIt, KeepIt)> for (A1, A2, A3)
    where A1: CastIndex<B, Dis> {}

    unsafe impl<A1, A2, A3, B, Dis> CastIndex<(A1, B, A3), (KeepIt, CastIt<Dis>, KeepIt)> for (A1, A2, A3)
    where A2: CastIndex<B, Dis> {}

    unsafe impl<A1, A2, A3, B, Dis> CastIndex<(A1, A2, B), (KeepIt, KeepIt, CastIt<Dis>)> for (A1, A2, A3)
    where A3: CastIndex<B, Dis> {}

    unsafe impl<'a, A: ? Sized, B: ? Sized, Dis: ? Sized> CastIndex<&'a B, &'a Dis> for &'a A
    where A: CastIndex<B, Dis> {}

    unsafe impl<'a, A: ? Sized, B: ? Sized, Dis: ? Sized> CastIndex<&'a mut B, &'a mut Dis> for &'a mut A
    where A: CastIndex<B, Dis> {}

    unsafe impl<A, B, Dis> CastIndex<[B], [Dis]> for [A]
    where A: CastIndex<B, Dis> {}

    unsafe impl<A, B, Dis> CastIndex<Vec<B>, Vec<Dis>> for Vec<A>
    where A: CastIndex<B, Dis> {}
}

mod collections {
    use super::*;
    use ::std::collections::{HashMap, BTreeMap};
    use ::std::collections::{HashSet, BTreeSet};

    unsafe impl<A, B, V, Dis> CastIndex<BTreeMap<B, V>, BTreeMap<CastIt<Dis>, KeepIt>> for BTreeMap<A, V>
    where A: CastIndex<B, Dis> {}

    unsafe impl<A, B, K, Dis> CastIndex<BTreeMap<K, B>, BTreeMap<KeepIt, CastIt<Dis>>> for BTreeMap<K, A>
    where A: CastIndex<B, Dis> {}

    unsafe impl<A, B, V, Dis> CastIndex<HashMap<B, V>, HashMap<CastIt<Dis>, KeepIt>> for HashMap<A, V>
    where A: CastIndex<B, Dis> {}

    unsafe impl<A, B, K, Dis> CastIndex<HashMap<K, B>, HashMap<KeepIt, CastIt<Dis>>> for HashMap<K, A>
    where A: CastIndex<B, Dis> {}

    unsafe impl<A, B, Dis> CastIndex<HashSet<B>, HashSet<Dis>> for HashSet<A>
    where A: CastIndex<B, Dis> {}

    unsafe impl<A, B, Dis> CastIndex<BTreeSet<B>, BTreeSet<Dis>> for BTreeSet<A>
    where A: CastIndex<B, Dis> {}
}

#[test]
fn test_it() {
    newtype_index!(A);
    newtype_index!(B);
    newtype_index!(C);

    let x: &usize = &0;
    let _: &A = cast_index(x);

    // explicit disambiguator.  All instances of KeepIt can be inferred.
    let x: &[(f64, Vec<(B, A, f64)>)] = &[];
    let _: &[(f64, Vec<(B, C, f64)>)] = cast_index::<_, _, &[(_, CastIt<Vec<(_, CastIt<_>, _)>>)]>(x);

    // actually, that disambiguator was not necessary.
    let x: &[(f64, Vec<(B, A, f64)>)] = &[];
    let _: &[(f64, Vec<(B, C, f64)>)] = cast_index(x);
}
