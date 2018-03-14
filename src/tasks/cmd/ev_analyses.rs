use ::ui::color::{ColorByRange, PaintAs, NullPainter};
use ::ui::cfg_merging::{merge_summaries, make_nested_mapping, no_summary};
use ::util::zip_eq;
use ::types::basis::Basis3;
use ::math::bands::{GammaUnfolder, ScMatrix};
use ::std::rc::Rc;
use ::std::mem;
use ::traits::alternate::{FnOnce, FnMut, Fn, StdFnMut, CallT};
use ::std::cell::RefCell;
use ::rsp2_tasks_config::Settings;
use ::errors::{StdResult, Result, ok};
#[allow(unused)] // compiler bug
use ::itertools::Itertools;

#[allow(unused)] // compiler bug
use ::rsp2_structure::{CoordStructure, Part, Partition};

use ::std::fmt;
use ::serde_yaml::Value as YamlValue;

mod cached {
    use super::*;
    use ::traits::alternate::{FnOnce};
    use self::inductive::hlist::{HList1, HNil};

    // NOTE: All clones must be tied to the same cache.
    pub trait Cached: Clone {
        type Value;

        fn call_cached(&self) -> Rc<Self::Value>;
    }

    pub struct Constant<A>(Rc<A>);
    impl<A> Clone for Constant<A> {
        fn clone(&self) -> Self { Constant(self.0.clone()) }
    }

    impl<A> Constant<A> {
        pub fn new(a: A) -> Self { Constant(Rc::new(a)) }
    }

    impl<A> Cached for Constant<A> {
        type Value = A;
        fn call_cached(&self) -> Rc<A> { self.0.clone() }
    }

    pub use self::function::Function;
    mod function {
        use super::*;

        pub struct Function<R, F>(Rc<RefCell<Inner<R, F>>>);
        impl<R, F> Clone for Function<R, F> {
            fn clone(&self) -> Self { Function(self.0.clone()) }
        }

        enum Inner<R, F> {
            Result(Rc<R>),
            Func(F),
            MidComputation,
        }

        impl<R, F> Cached for Function<R, F> where F: FnOnce<HNil, Output=Rc<R>> {
            type Value = R;

            fn call_cached(&self) -> Rc<R> {
                let mut inner = self.0.borrow_mut();
                let result = match mem::replace(&mut *inner, Inner::MidComputation) {
                    Inner::MidComputation => panic!("detected cyclic dependency in Cached objects!"),
                    Inner::Func(f) => f.call_once(HNil),
                    Inner::Result(r) => r,
                };
                *inner = Inner::Result(result.clone());
                result
            }
        }
    }

    pub use self::map::Map;
    mod map {
        use super::*;

        pub struct Map<A, R, F>(Function<R, Closure<A, F>>);
        impl<A, R, F> Clone for Map<A, R, F> {
            fn clone(&self) -> Self { Map(self.0.clone()) }
        }

        struct Closure<A, F> {
            arg: A,
            continuation: F,
        }

        impl<W, A, R, F> FnOnce<HNil> for Closure<A, F>
        where A: Cached<Value=W>, F: FnOnce<HList1<Rc<W>>, Output=Rc<R>>,
        {
            type Output = Rc<R>;

            fn call_once(self, HNil: HNil) -> Rc<R>
            { self.continuation.call_once(hlist![self.arg.call_cached()]) }
        }

        impl<W, A, R, F> Cached for Map<A, R, F>
        where A: Cached<Value=W>, F: FnOnce<HList1<Rc<W>>, Output=Rc<R>>,
        {
            type Value = R;

            fn call_cached(&self) -> Rc<R>
            { self.0.call_cached() }
        }
    }

    pub use self::and_then::AndThen;
    mod and_then {
        use super::*;

        pub struct AndThen<A, R, F>(Function<R, Closure<A, F>>);
        impl<A, R, F> Clone for AndThen<A, R, F> {
            fn clone(&self) -> Self { AndThen(self.0.clone()) }
        }

        struct Closure<A, F> {
            arg: A,
            continuation: F,
        }

        impl<D, W, A, R, F> FnOnce<HNil> for Closure<A, F>
        where A: Cached<Value=W>, F: FnOnce<HList1<Rc<W>>, Output=D>, D: Cached<Value=R>,
        {
            type Output = Rc<R>;

            fn call_once(self, HNil: HNil) -> Rc<R>
            { self.continuation.call_once(hlist![self.arg.call_cached()]).call_cached() }
        }

        impl<D, W, A, R, F> Cached for AndThen<A, R, F>
        where A: Cached<Value=W>, F: FnOnce<HList1<Rc<W>>, Output=D>, D: Cached<Value=R>,
        {
            type Value = R;

            fn call_cached(&self) -> Rc<R>
            { self.0.call_cached() }
        }
    }
}

mod closures {
    use super::*;
    use self::inductive::hlist::{HList, HNil, HCons, HList1, HList2, h_cons};

    #[derive(Debug, Copy, Clone, Default)]
    pub struct Id;
    derive_alternate_fn! {
        impl[X] Fn<Hlist![X]> for Id {
            type Output = X;

            fn call(&self, hlist_pat![x]: Hlist![X]) -> X
            { x }
        }
    }

    /// Variadic function that puts all its arguments into one HList.
    ///
    /// This function has no inverse, since a function can only have one output.
    /// (instead, you must use an HOF adapter like `OnHLists`)
    #[derive(Debug, Copy, Clone, Default)]
    pub struct MakeHList;
    derive_alternate_fn! {
        impl[X: HList] Fn<X> for MakeHList {
            type Output = X;

            fn call(&self, x: X) -> X
            { x }
        }
    }

    #[allow(unused)]
    pub type SwapT<List> = <Swap as FnOnce<List>>::Output;

    /// Swap the first two elements of an HList.
    #[derive(Debug, Copy, Clone, Default)]
    pub struct Swap;
    derive_alternate_fn! {
        impl[A, B, Rest: HList] Fn<HList1<HCons<A, HCons<B, Rest>>>> for Swap {
            type Output = HCons<B, HCons<A, Rest>>;

            fn call(&self, hlist_pat![list]: HList1<HCons<A, HCons<B, Rest>>>) -> Self::Output {
                let (a, list) = list.pop();
                let (b, list) = list.pop();
                h_cons(b, h_cons(a, list))
            }
        }
    }

    /// Make a function of multiple arguments act on one HList.
    ///
    /// Similar in spirit to Haskell's `uncurry`.
    #[derive(Debug, Copy, Clone, Default)]
    pub struct OnHLists<F>(pub F);

    /// Inverse of `OnHLists`. Make a function of HList act on multiple arguments.
    pub type UnHLists<F> = Of<F, MakeHList>;
    /// Fake constructor for `UnHLists`.
    #[allow(bad_style)]
    pub fn UnHLists<F>(f: F) -> UnHLists<F> { Of(f, MakeHList) }

    impl<F, L: HList> Fn<HList1<L>> for OnHLists<F>
    where F: Fn<L>,
    {
        fn call(&self, hlist_pat![list]: HList1<L>) -> Self::Output
        { self.0.call(list) }
    }

    impl<F, L: HList> FnMut<HList1<L>> for OnHLists<F>
    where F: FnMut<L>,
    {
        fn call_mut(&mut self, hlist_pat![list]: HList1<L>) -> Self::Output
        { self.0.call_mut(list) }
    }

    impl<F, L: HList> FnOnce<HList1<L>> for OnHLists<F>
    where F: FnOnce<L>,
    {
        type Output = F::Output;

        fn call_once(self, hlist_pat![list]: HList1<L>) -> Self::Output
        { self.0.call_once(list) }
    }

    /// Flip the first two arguments of a function.
    ///
    /// Even though this is just a type alias, you can construct one
    /// by writing `Flip(f)`.  (this is because in reality, there is
    /// also a free function with the same name)
    pub type Flip<F> = UnHLists<Of<OnHLists<F>, Swap>>;

    /// Constructor for the type `Flip<F>`.
    #[allow(bad_style)]
    pub fn Flip<F>(f: F) -> Flip<F>
    { UnHLists(Of(OnHLists(f), Swap)) }

      #[allow(unused)]
    fn test_on_hlist() {
        assert_eq!(OnHLists(|a, b| a + b).call_once(hlist![hlist![1, 5]]), 6);
        assert_eq!(UnHLists(|hlist_pat![a, b]| a + b).call_once(hlist![1, 5]), 6);

        let v = vec![3];
        assert_eq!(OnHLists(|| v).call_once(hlist![hlist![]]), vec![3]);
    }

    #[allow(unused)]
    fn test_swap() {
        assert_eq!(Swap.call_once(hlist![hlist![1, 5]]), hlist![5, 1]);
        assert_eq!(Swap.call_once(hlist![hlist![1, 5, 4]]), hlist![5, 1, 4]);
    }

    #[allow(unused)]
    fn test_flip() {
        assert_eq!(Fn::call(&Flip(|a, b|     a - b     ), hlist![1, 5]   ),  4);
        assert_eq!(Fn::call(&Flip(|a, b, c| (a - b) * c), hlist![1, 5, 7]), 28);
    }

    #[allow(unused)]
    pub type OfT<F, G, Args> = <Of<F, G> as FnOnce<Args>>::Output;
    #[derive(Debug, Copy, Clone, Default)]
    pub struct Of<F, G>(pub F, pub G);
    impl<Args: HList, B, C, F, G> Fn<Args> for Of<F, G>
    where
        G: Fn<Args, Output=B>,
        F: Fn<HList1<B>, Output=C>,
    {
        fn call(&self, args: Args) -> C
        { self.0.call(hlist![self.1.call(args)]) }
    }

    impl<Args: HList, B, C, F, G> FnMut<Args> for Of<F, G>
    where
        G: FnMut<Args, Output=B>,
        F: FnMut<HList1<B>, Output=C>,
    {
        fn call_mut(&mut self, args: Args) -> C
        { self.0.call_mut(hlist![self.1.call_mut(args)]) }
    }

    impl<Args: HList, B, C, F, G> FnOnce<Args> for Of<F, G>
    where
        G: FnOnce<Args, Output=B>,
        F: FnOnce<HList1<B>, Output=C>,
    {
        type Output = C;

        fn call_once(self, args: Args) -> C
        { self.0.call_once(hlist![self.1.call_once(args)]) }
    }

    #[allow(unused)]
    fn test_of() {
        struct A; struct B; struct C;
        struct F; struct G;
        impl FnOnce<HList1<A>> for F {
            type Output = B;
            fn call_once(self, _: HList1<A>) -> B { B }
        }
        impl FnOnce<HList1<B>> for G {
            type Output = C;
            fn call_once(self, _: HList1<B>) -> C { C }
        }
        let _: B = F.call_once(hlist![A]);
        let _: C = G.call_once(hlist![B]);
        let _: C = Of(G, F).call_once(hlist![A]);
    }
}

/// Inductive types, to assist traits that simulate variadic generics.
pub mod inductive {
    use super::*;

    pub use self::peano::{P0, Succ, IsPeano, Pred};
    pub mod peano {
        use super::*;

        pub use self::constants::*;
        pub mod constants {
            use super::*;
            pub use super::P0;
            #[doc = "Peano integer."] pub type P1 = Succ<P0>;
            #[doc = "Peano integer."] pub type P2 = Succ<P1>;
            #[doc = "Peano integer."] pub type P3 = Succ<P2>;
            #[doc = "Peano integer."] pub type P4 = Succ<P3>;
            #[doc = "Peano integer."] pub type P5 = Succ<P4>;
            #[doc = "Peano integer."] pub type P6 = Succ<P5>;
            #[doc = "Peano integer."] pub type P7 = Succ<P6>;
            #[doc = "Peano integer."] pub type P8 = Succ<P7>;
            #[doc = "Peano integer."] pub type P9 = Succ<P8>;
        }

        /// The Peano integer zero. (base case)
        #[derive(Debug, Default, Copy, Clone)]
        pub struct P0;

        /// A Peano integer incremented by 1. (inductive case)
        #[derive(Debug, Default, Copy, Clone)]
        pub struct Succ<A: IsPeano>(pub A);

        /// Marker trait for Peano integers, for "type safety" within the type system.
        pub trait IsPeano {}
        impl IsPeano for P0 {}
        impl<N: IsPeano> IsPeano for Succ<N> {}

        /// Subtract 1 from a nonzero Peano integer.
        ///
        /// More accurately, this solves the equation `p + 1 = x` for `p`.
        pub type Pred<A> = <A as Positive>::Pred;

        pub trait Positive: IsPeano {
            type Pred: IsPeano;
        }

        impl<N: IsPeano> Positive for Succ<N> {
            type Pred = N;
        }
    }

    pub mod hlist {
        use super::*;

        pub use frunk::hlist::{HList, HCons, HNil, h_cons};

        // wake me up when clion can parse type macros properly
        pub type HList1<A>          = HCons<A, HNil>;
        pub type HList2<A, B>       = HCons<A, HList1<B>>;
        pub type HList3<A, B, C>    = HCons<A, HList2<B, C>>;
        pub type HList4<A, B, C, D> = HCons<A, HList3<B, C, D>>;

        //----------------

        #[doc(hidden)]
        pub type MapT<List, F> = <List as Map<F>>::Output;

        pub trait Map<F>: HList {
            type Output: HList;

            fn map(self, f: F) -> Self::Output;
        }

        impl<F> Map<F> for HNil {
            type Output = HNil;

            #[inline(always)]
            fn map(self, _: F) -> Self::Output
            { HNil }
        }

        impl<A, B, Rest: Map<F>, F> Map<F> for HCons<A, Rest>
        where F: FnMut<HList1<A>, Output=B>,
        {
            type Output = HCons<B, MapT<Rest, F>>;

            #[inline(always)]
            fn map(self, mut f: F) -> Self::Output {
                let (a, rest) = self.pop();
                h_cons(f.call_mut(hlist![a]), rest.map(f))
            }
        }

        //----------------

        pub struct HConsClosure;
        derive_alternate_fn! {
            impl[A, List: HList] Fn<Hlist![A, List]> for HConsClosure {
                type Output = HCons<A, List>;

                fn call(&self, hlist_pat![a, list]: Hlist![A, List]) -> Self::Output
                { h_cons(a, list) }
            }
        }


        pub struct ReverseClosure;
        derive_alternate_fn! {
            impl[List: ::frunk::hlist::IntoReverse] Fn<Hlist![List]> for hlist::ReverseClosure {
                type Output = List::Output;

                fn call(&self, hlist_pat![list]: Hlist![List]) -> Self::Output
                { list.into_reverse() }
            }
        }

        #[allow(unused)]
        fn test_reverse() {
            struct A; struct B; struct C;
            let hlist_pat![] = ReverseClosure.call_once(hlist![hlist![]]);
            let hlist_pat![A] = ReverseClosure.call_once(hlist![hlist![A]]);
            let hlist_pat![B, A] = ReverseClosure.call_once(hlist![hlist![A, B]]);
            let hlist_pat![C, B, A] = ReverseClosure.call_once(hlist![hlist![A, B, C]]);
        }
    }
}

pub mod option {
    use super::*;
    use ::std::ops::{Deref, DerefMut};
    use ::std::marker::PhantomData;
    use self::inductive::hlist::{self, HNil, HCons, HList, HList1, HList2};

    #[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Default)]
    pub struct Just<T>(pub T);
    #[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
    pub struct Nothing<T>(pub PhantomData<T>);
    pub type Maybe<T> = Option<T>;

    // for type-level type-checking
    pub trait IsMaybe { }
    impl<T> IsMaybe for Just<T> { }
    impl<T> IsMaybe for Maybe<T> { }
    impl<T> IsMaybe for Nothing<T> { }

    impl<T> Deref for Just<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target
        { &self.0 }
    }

    impl<T> DerefMut for Just<T> {
        fn deref_mut(&mut self) -> &mut Self::Target
        { &mut self.0 }
    }

    impl<T> Default for Nothing<T> { fn default() -> Self { Nothing(Default::default()) } }
    impl<T> Clone for Nothing<T> { fn clone(&self) -> Self { Default::default() } }
    impl<T> Copy for Nothing<T> { }

    #[allow(unused)]
    pub type MapT<A, F> = <A as Map<F>>::Output;
    pub trait Map<F>: Sized + IsMaybe {
        type Output;

        fn map(self, f: F) -> Self::Output;
    }

    impl<A, B, F: FnOnce<HList1<A>, Output=B>> Map<F> for Just<A> {
        type Output = Just<B>;

        fn map(self, f: F) -> Self::Output
        { Just(f.call_once(hlist![self.0])) }
    }

    impl<A, B, F: FnOnce<HList1<A>, Output=B>> Map<F> for Maybe<A> {
        type Output = Maybe<B>;

        fn map(self, f: F) -> Self::Output
        { Option::map(self, |x| f.call_once(hlist![x])) }
    }

    impl<A, B, F: FnOnce<HList1<A>, Output=B>> Map<F> for Nothing<A> {
        type Output = Nothing<B>;

        fn map(self, _: F) -> Self::Output
        { Default::default() }
    }

//    #[allow(unused)]
//    pub type ZipT<A, B> = ZipWithT<closures::MakeTuple, A, B>;
//    pub trait Zip<Rhs: IntoMaybe>: ZipWith<Rhs, closures::MakeTuple> {
//        fn zip(self, other: Rhs) -> Self::Output;
//    }
//    impl<A, B: IntoMaybe> Zip<B> for A where A: ZipWith<B, closures::MakeTuple> {
//        fn zip(self, other: B) -> Self::Output
//        { self.zip_with(other, closures::MakeTuple) }
//    }

    #[allow(unused)]
    pub type ZipWithT<F, A, B> = <A as ZipWith<B, F>>::Output;
    pub trait ZipWith<Rhs: IntoMaybe, F>: Sized + IntoMaybe {
        type Output;

        fn zip_with(self, other: Rhs, f: F) -> Self::Output;
    }

    pub use self::list_zip::{ListZip, ListZipT};
    mod list_zip {
        use super::*;

        #[allow(unused)]
        pub type ListZipT<List> = <List as ListZip>::Output;
        pub trait ListZip: HList {
            type Output: IsMaybe;

            fn list_zip(self) -> Self::Output;
        }

        impl<List: Impl> ListZip for List
        where
            ImplT<Self>: Map<hlist::ReverseClosure>,
            MapT<ImplT<Self>, hlist::ReverseClosure>: IsMaybe,
        {
            type Output = MapT<ImplT<Self>, hlist::ReverseClosure>;

            fn list_zip(self) -> Self::Output
            {
                let maybe_reversed = self.rec(Just(HNil));
                maybe_reversed.map(hlist::ReverseClosure)
            }
        }

        pub type ImplInit = Just<HNil>;
        pub type ImplT<List> = <List as Impl>::Output;
        pub trait Impl<Acc: IsMaybe = ImplInit>: HList {
            type Output: IsMaybe;

            fn rec(self, acc: Acc) -> Self::Output;
        }

        impl<Acc: IsMaybe> Impl<Acc> for HNil {
            type Output = Acc;

            fn rec(self, acc: Acc) -> Self::Output { acc }
        }

        impl<
            A: IntoMaybe<TyArg=AInner> + ZipWith<Acc, hlist::HConsClosure, Output=Z>,
            Acc: IntoMaybe<TyArg=AccInner>,
            AccInner,
            AInner,
            Z: IsMaybe,
            Rest: Impl<Z, Output=R>,
            R: IsMaybe,
        > Impl<Acc> for HCons<A, Rest>
        {
            type Output = R;

            fn rec(self, acc: Acc) -> Self::Output
            { self.tail.rec(self.head.zip_with(acc, hlist::HConsClosure)) }
        }
    }

    #[allow(unused)]
    pub type AsRefT<A> = <A as AsRef>::Output;
    pub trait AsRef {
        type Output;

        fn as_ref(self) -> Self::Output;
    }

    impl<'a, A> AsRef for &'a Maybe<A> {
        type Output = Maybe<&'a A>;

        fn as_ref(self) -> Self::Output
        { Option::as_ref(self) }
    }

    impl<'a, A> AsRef for &'a Just<A> {
        type Output = Just<&'a A>;

        fn as_ref(self) -> Self::Output
        { Just(&self.0) }
    }

    impl<'a, A> AsRef for &'a Nothing<A> {
        type Output = Nothing<&'a A>;

        fn as_ref(self) -> Self::Output
        { Default::default() }
    }

    #[derive(Debug, Copy, Clone, Default)]
    pub struct AsRefClosure;
    derive_alternate_fn! {
        impl[A: AsRef] Fn<HList1<A>> for AsRefClosure {
            type Output = AsRefT<A>;

            fn call(&self, hlist_pat![a]: HList1<A>) -> Self::Output
            { a.as_ref() }
        }
    }

    pub trait IntoMaybe: IsMaybe {
        type TyArg;

        fn into_maybe(self) -> Maybe<Self::TyArg>;
        fn as_maybe(&self) -> Maybe<&Self::TyArg>;
    }

    impl<A> IntoMaybe for Just<A> {
        type TyArg = A;

        fn into_maybe(self) -> Maybe<A> { Some(self.0) }
        fn as_maybe(&self) -> Maybe<&A> { Some(&self.0) }
    }

    impl<A> IntoMaybe for Maybe<A> {
        type TyArg = A;

        fn into_maybe(self) -> Maybe<A> { self }
        fn as_maybe(&self) -> Maybe<&A> { self.as_ref() }
    }

    impl<A> IntoMaybe for Nothing<A> {
        type TyArg = A;

        fn into_maybe(self) -> Maybe<A> { None }
        fn as_maybe(&self) -> Maybe<&A> { None }
    }

    pub trait ExpectFromOption<B>: IsMaybe {
        fn expect_from(b: Option<B>) -> Self;
    }

    impl<A> ExpectFromOption<A> for Maybe<A> {
        fn expect_from(a: Option<A>) -> Self { a }
    }

    impl<A> ExpectFromOption<A> for Just<A> {
        fn expect_from(a: Option<A>) -> Self { Just(a.unwrap()) }
    }

    impl<A> ExpectFromOption<A> for Nothing<A> {
        fn expect_from(a: Option<A>) -> Self
        { match a {
            Some(_) => panic!("expect_from: expected nothing, got Some(_)!"),
            None => Default::default(),
        }}
    }

    #[allow(unused)]
    pub type FoldOkT<A> = <A as FoldOk>::Output;
    pub trait FoldOk {
        type Output;

        fn fold_ok(self) -> Self::Output;
    }

    impl<A, E> FoldOk for Maybe<StdResult<A, E>> {
        type Output = StdResult<Maybe<A>, E>;

        fn fold_ok(self) -> Self::Output
        { match self {
            Some(x) => Ok(Some(x?)),
            None => Ok(None),
        }}
    }

    impl<A, E> FoldOk for Just<StdResult<A, E>> {
        type Output = StdResult<Just<A>, E>;

        fn fold_ok(self) -> Self::Output
        { Ok(Just(self.0?)) }
    }


    impl<A, E> FoldOk for Nothing<StdResult<A, E>> {
        type Output = StdResult<Nothing<A>, E>;

        fn fold_ok(self) -> Self::Output
        { Ok(Default::default()) }
    }

    macro_rules! impl_zip_with {
        ( $( [$A:ident, $B:ident] => $C:ident;)* )
        => {
            $(
                impl<A, B, C, F: FnOnce<HList2<A, B>, Output=C>> ZipWith<$B<B>, F> for $A<A>
                where F: FnOnce<HList2<A, B>>,
                {
                    type Output = $C<C>;

                    fn zip_with(self, other: $B<B>, f: F) -> Self::Output {
                        ExpectFromOption::expect_from({
                            match (self.into_maybe(), other.into_maybe()) {
                                (Some(a), Some(b)) => Some(f.call_once(hlist![a, b])),
                                _ => None,
                            }
                        })
                    }
                }
            )*
        };
    }

    impl_zip_with!{
        [Nothing, Nothing] => Nothing;
        [  Maybe, Nothing] => Nothing;
        [   Just, Nothing] => Nothing;
        [Nothing,   Maybe] => Nothing;
        [Nothing,    Just] => Nothing;
        [  Maybe,   Maybe] =>   Maybe;
        [  Maybe,    Just] =>   Maybe;
        [   Just,   Maybe] =>   Maybe;
        [   Just,    Just] =>    Just;
    }
}


#[derive(Debug, Clone)] pub struct AtomCoordinates(pub CoordStructure);
#[derive(Debug, Clone)] pub struct AtomLayers(pub Vec<usize>);
#[derive(Debug, Clone)] pub struct LayerScMatrices(pub Vec<ScMatrix>);
#[derive(Debug, Clone)] pub struct EvFrequencies(pub Vec<f64>);
#[derive(Debug, Clone)] pub struct EvEigenvectors(pub Basis3);

pub use self::gamma_system_analysis::GammaSystemAnalysis;
pub mod gamma_system_analysis {
    use super::*;
    use self::inductive::hlist::{self, HList};

    pub trait Analyze<F> {
        type Output;

        fn analyze(self, f: F) -> Self::Output;
    }

    impl<'a, F, Z, M, As: HList, Rs: HList, Out> Analyze<As> for F
    where
        As: hlist::Map<option::AsRefClosure, Output=Rs>,
        Rs: option::ListZip<Output=Z>,
        Z: option::Map<closures::OnHLists<F>, Output=M>,
        M: option::FoldOk<Output=Out>,
    {
        type Output = Out;

        fn analyze(self, args: As) -> Self::Output {
            args.map(option::AsRefClosure)
                .list_zip()
                .map(closures::OnHLists(self))
                .fold_ok()
        }
    }

    pub struct Input<'a, A, B, C, D, E>
    where A: 'a, B: 'a, C: 'a, D: 'a, E: 'a,
    {
        pub atom_coords:     &'a A,
        pub atom_layers:     &'a B,
        pub layer_sc_mats:   &'a C,
        pub ev_frequencies:  &'a D,
        pub ev_eigenvectors: &'a E,
    }

    pub struct GammaSystemAnalysis<A, B, C, D, E>
    where
    {
        pub ev_frequencies:        A,
        pub ev_acousticness:       B,
        pub ev_polarization:       C,
        pub ev_layer_gamma_probs:  D,
        pub ev_layer_acousticness: E,
    }

    impl<
        'a,
        InAtomCoordinates,
        InAtomLayers,
        InLayerScMatrices,
        InEvFrequencies,
        InEvEigenvectors,
    > Input<
        'a,
        InAtomCoordinates,
        InAtomLayers,
        InLayerScMatrices,
        InEvFrequencies,
        InEvEigenvectors,
    > {
        pub fn compute<
            OutEvAcousticness,
            OutEvPolarization,
            OutEvLayerGammaProbs,
            OutEvLayerAcousticness,
        >(&self) -> Result<GammaSystemAnalysis<
            InEvFrequencies,
            OutEvAcousticness,
            OutEvPolarization,
            OutEvLayerGammaProbs,
            OutEvLayerAcousticness,
        >>
        where
            InEvFrequencies: Clone,
            ev_acousticness::Analysis: Analyze<hlist::HList1<&'a InEvEigenvectors>, Output=Result<OutEvAcousticness>>,
            ev_polarization::Analysis: Analyze<hlist::HList1<&'a InEvEigenvectors>, Output=Result<OutEvPolarization>>,
            ev_layer_gamma_probs::Analysis: Analyze<hlist::HList4<&'a InAtomLayers, &'a InAtomCoordinates, &'a InLayerScMatrices, &'a InEvEigenvectors>, Output=Result<OutEvLayerGammaProbs>>,
            ev_layer_acousticness::Analysis: Analyze<hlist::HList2<&'a InAtomLayers, &'a InEvEigenvectors>, Output=Result<OutEvLayerAcousticness>>,
        {ok({
            let Input {
                atom_coords, atom_layers, layer_sc_mats,
                ev_frequencies, ev_eigenvectors,
            } = *self;

            let ev_acousticness = ev_acousticness::Analysis.analyze(hlist![
                ev_eigenvectors,
            ])?;

            let ev_polarization = ev_polarization::Analysis.analyze(hlist![
                ev_eigenvectors,
            ])?;

            let ev_layer_gamma_probs = ev_layer_gamma_probs::Analysis.analyze(hlist![
                atom_layers, atom_coords, layer_sc_mats, ev_eigenvectors,
            ])?;

            let ev_layer_acousticness = ev_layer_acousticness::Analysis.analyze(hlist![
                atom_layers, ev_eigenvectors,
            ])?;

            let ev_frequencies = ev_frequencies.clone();

            GammaSystemAnalysis {
                ev_frequencies,
                ev_acousticness,
                ev_polarization,
                ev_layer_gamma_probs,
                ev_layer_acousticness,
            }
        })}
    }
}

macro_rules! wrap_maybe_compute {
    (
        $(
            pub $struct_or_enum:tt $Thing:ident
                $(   { $($Thing_body_if_brace:tt)* }      )*
                $($( ( $($Thing_body_if_paren:tt)*  ) )* ; )*

            fn $thing:ident(
                $($arg:ident : & $Arg:ident),* $(,)*
            ) -> Result<_>
            $fn_body:block
        )*
    ) => {
        $(
            pub use self::$thing::$Thing;
            pub mod $thing {
                use super::*;

                pub $struct_or_enum $Thing
                $(   { $($Thing_body_if_brace)* }      )*
                $($( ( $($Thing_body_if_paren)* ) )* ; )*

                pub struct Analysis;
                impl<'a> FnOnce<Hlist![$(&'a $Arg,)*]> for Analysis {
                    type Output = Result<$Thing>;

                    fn call_once(self, hlist_pat![$($arg),*] : Hlist![$(&'a $Arg),*]) -> Self::Output
                    $fn_body
                }
            }
        )*
    }
}

// Example of what the macro expands into.
pub use self::ev_acousticness::EvAcousticness;
pub mod ev_acousticness {
    use super::*;

    use self::inductive::hlist::HList1;

    pub struct EvAcousticness(pub Vec<f64>);
    pub struct Analysis;
    impl<'a> FnOnce<HList1<&'a EvEigenvectors>> for Analysis {
        type Output = Result<EvAcousticness>;

        fn call_once(self, hlist_pat![ev_eigenvectors]: HList1<&EvEigenvectors>) -> Self::Output {
            Ok(EvAcousticness((ev_eigenvectors.0).0.iter().map(|ket| ket.acousticness()).collect()))
        }
    }
}

wrap_maybe_compute! {
    pub struct EvPolarization(pub Vec<[f64; 3]>);
    fn ev_polarization(ev_eigenvectors: &EvEigenvectors) -> Result<_> {
        Ok(EvPolarization((ev_eigenvectors.0).0.iter().map(|ket| ket.polarization()).collect()))
    }
}

wrap_maybe_compute! {
    pub struct EvLayerAcousticness(pub Vec<f64>);
    fn ev_layer_acousticness(
        atom_layers: &AtomLayers,
        ev_eigenvectors: &EvEigenvectors,
    ) -> Result<_> {
        let part = Part::from_ord_keys(atom_layers.0.iter());
        Ok(EvLayerAcousticness({
            (ev_eigenvectors.0).0.iter().map(|ket| {
                ket.clone()
                    .into_unlabeled_partitions(&part)
                    .map(|ev| ev.acousticness())
                    .sum()
            }).collect()
        }))
    }
}

wrap_maybe_compute! {
    pub struct EvLayerGammaProbs(pub Vec<Vec<f64>>);
    fn ev_layer_gamma_probs(
        atom_layers: &AtomLayers,
        atom_coords: &AtomCoordinates,
        layer_sc_mats: &LayerScMatrices,
        ev_eigenvectors: &EvEigenvectors,
    ) -> Result<_> {
        let part = Part::from_ord_keys(atom_layers.0.iter());
        let coords_by_layer = atom_coords.0
            .map_metadata_to(|_| ())
            .into_unlabeled_partitions(&part)
            .collect_vec();

        let evs_by_layer = (ev_eigenvectors.0).0.iter().map(|ket| {
            ket.clone().into_unlabeled_partitions(&part)
        });
        let evs_by_layer = ::util::transpose_iter_to_vec(evs_by_layer);

        Ok(EvLayerGammaProbs({
            zip_eq(coords_by_layer, zip_eq(evs_by_layer, &layer_sc_mats.0))
                .map(|(layer_structure, (layer_evs, layer_sc_mat))| {
                    // precompute data applicable to all kets
                    let unfolder = GammaUnfolder::from_config(
                        &from_json!({
                            "fbz": "reciprocal-cell",
                            "sampling": { "plain": [4, 4, 1] },
                        }),
                        &layer_structure,
                        layer_sc_mat,
                    );

                    layer_evs.into_iter().map(|ket| {
                        let probs = unfolder.unfold_phonon(ket.to_ket().as_ref());
                        zip_eq(unfolder.q_indices(), probs)
                            .find( | & (idx, _) | idx == & [0, 0, 0])
                            .unwrap().1
                    }).collect()
                }).collect()
        }))
    }
}

macro_rules! format_columns {
    (
        $header_fmt: expr,
        $entry_fmt: expr,
        $columns1:ident $(,)*
    ) => {
        Columns {
            header: format!($header_fmt, $columns1.header),
            entries: {
                $columns1.entries.into_iter()
                    .map(|$columns1| format!($entry_fmt, $columns1))
                    .collect()
            },
        }
    };
    (
        $header_fmt: expr,
        $entry_fmt: expr,
        $columns1:ident, $($columns:ident,)+ $(,)*
    ) => {
        Columns {
            header: format!($header_fmt, $columns1.header, $($columns.header),+),
            entries: {
                izip!($columns1.entries, $(&$columns.entries),+)
                    .map(|($columns1, $($columns),*)| format!($entry_fmt, $columns1, $($columns),*))
                    .collect()
            },
        }
    };
}

pub enum ColumnsMode {
    ForHumans,
    ForMachines,
}

impl<A, B, C, D, E> GammaSystemAnalysis<A, B, C, D, E>
where
    A: option::IntoMaybe<TyArg=EvFrequencies>,
    B: option::IntoMaybe<TyArg=EvAcousticness>,
    C: option::IntoMaybe<TyArg=EvPolarization>,
    D: option::IntoMaybe<TyArg=EvLayerGammaProbs>,
    E: option::IntoMaybe<TyArg=EvLayerAcousticness>,
{
    pub fn make_columns(&self, mode: ColumnsMode) -> Option<Columns> {
        let &GammaSystemAnalysis {
            ref ev_frequencies,
            ref ev_acousticness,
            ref ev_polarization,
            ref ev_layer_gamma_probs,
            ref ev_layer_acousticness,
        } = self;
        let ev_frequencies = ev_frequencies.as_maybe();
        let ev_acousticness = ev_acousticness.as_maybe();
        let ev_polarization = ev_polarization.as_maybe();
        let ev_layer_gamma_probs = ev_layer_gamma_probs.as_maybe();
        let ev_layer_acousticness = ev_layer_acousticness.as_maybe();

        use self::Color::{Colorful, Colorless};

        let mut columns = vec![];

        let fix1 = |c, title: &str, data: &_| fixed_prob_column(c, Precision(1), title, data);
        let fix2 = |c, title: &str, data: &_| fixed_prob_column(c, Precision(2), title, data);
        let dp = display_prob_column;

        if let Some(ref data) = ev_frequencies {
            let col = Columns {
                header: "Frequency(cm-1)".into(),
                entries: data.0.to_vec(),
            };
            columns.push(format_columns!(
                "{:27}",
                "{:27}",
                col,
            ))
        };

        if let Some(ref data) = ev_acousticness {
            columns.push(match mode {
                ColumnsMode::ForHumans => dp(Colorful, "Acoust.", &data.0),
                ColumnsMode::ForMachines => fix2(Colorless, "Acou", &data.0),
            })
        };

        if let Some(ref data) = ev_layer_acousticness {
            columns.push(match mode {
                ColumnsMode::ForHumans => dp(Colorful, "Layer", &data.0),
                ColumnsMode::ForMachines => fix2(Colorless, "Lay.", &data.0),
            })
        };

        if let Some(ref data) = ev_polarization {
            let data = |k| data.0.iter().map(|v| v[k]).collect_vec();
            let name = |k| "XYZ".chars().nth(k).unwrap().to_string();

            columns.push(match mode {
                ColumnsMode::ForHumans => {
                    let axis = |k| fix2(Colorful, &name(k), &data(k));
                    let (x, y, z) = (axis(0), axis(1), axis(2));
                    format_columns! {
                        "[{}, {}, {}]",
                        "[{}, {}, {}]",
                        x, y, z,
                    }
                },
                ColumnsMode::ForMachines => {
                    let axis = |k| fix2(Colorless, &name(k), &data(k));
                    let (x, y, z) = (axis(0), axis(1), axis(2));
                    format_columns! {
                        "{} {} {}",
                        "{} {} {}",
                        x, y, z,
                    }
                },
            });
        }

        if let Some(ref data) = ev_layer_gamma_probs {
            for (n, probs) in data.0.iter().enumerate() {
                columns.push(match mode {
                    ColumnsMode::ForHumans   => fix1(Colorful,  &format!("G{:02}", n+1), &probs),
                    ColumnsMode::ForMachines => fix1(Colorless, &format!("G{:02}", n+1), &probs),
                })
            }
        }

        match columns.len() {
            0 => None,
            _ => Some({
                let joined = columns::join(&columns, " ");
                format_columns!(
                    "# {}", // "comment out" the header
                    "  {}",
                    joined,
                )
            }),
        }
    }

    pub fn make_summary(&self, settings: &Settings) -> YamlValue {
        let GammaSystemAnalysis {
            ref ev_acousticness, ref ev_polarization,
            ref ev_frequencies, ref ev_layer_gamma_probs,
            ref ev_layer_acousticness,
        } = *self;

        // This is where the newtypes start to get in the way;
        // turn as many things as we can into Option<Vec<T>> (indexed by ket)
        // for ease of composition.
        let frequency = ev_frequencies.as_maybe().map(|d| d.0.to_vec());
        let acousticness = ev_acousticness.as_maybe().map(|d| d.0.to_vec());
        let polarization = ev_polarization.as_maybe().map(|d| d.0.to_vec());
        let layer_acousticness = ev_layer_acousticness.as_maybe().map(|d| d.0.to_vec());

        // Work with Option<Vec<A>> as an applicative functor (for fixed length Vec)
        fn map1<A, R, F>(a: &Option<Vec<A>>, mut f: F) -> Option<Vec<R>>
        where F: for<'a> StdFnMut(&'a A) -> R
        {
            let a = a.as_ref()?;
            Some(a.iter().map(|a| f(a)).collect())
        }

        // (haskell's LiftA2)
        fn map2<B, A, R, F>(a: &Option<Vec<A>>, b: &Option<Vec<B>>, mut f: F) -> Option<Vec<R>>
        where F: for<'a, 'b> StdFnMut(&'a A, &'b B) -> R
        {
            Some({
                zip_eq(a.as_ref()?, b.as_ref()?)
                    .map(|(a, b)| f(a, b))
                    .collect()
            })
        }

        // form a miniature DSL to reduce chances of accidental
        // discrepancy between closure args and parameters (
        fn at_least(thresh: f64, a: &Option<Vec<f64>>) -> Option<Vec<bool>>
        { map1(a, |&a| thresh <= a) }

        fn and(a: &Option<Vec<bool>>, b: &Option<Vec<bool>>) -> Option<Vec<bool>>
        { map2(a, b, |&a, &b| a && b) }

        fn zip<A: Clone, B: Clone>(a: &Option<Vec<A>>, b: &Option<Vec<B>>) -> Option<Vec<(A, B)>>
        { map2(a, b, |a, b| (a.clone(), b.clone())) }

        fn not(a: &Option<Vec<bool>>) -> Option<Vec<bool>>
        { map1(a, |a| !a) }

        fn enumerate<T>(a: &Option<Vec<T>>) -> Option<Vec<(usize, &T)>>
        { a.as_ref().map(|a| a.iter().enumerate().collect()) }

        fn select<T: Clone>(pred: &Option<Vec<bool>>, values: &Option<Vec<T>>) -> Option<Vec<T>>
        {
            Some({
                zip_eq(pred.as_ref()?, values.as_ref()?)
                    .filter_map(|(&p, v)| if p { Some(v.clone()) } else { None })
                    .collect()
            })
        }

        let z_polarization = map1(&polarization, |p| p[2]);
        let xy_polarization = map1(&polarization, |p| p[0] + p[1]);

        let is_acoustic = at_least(0.95, &acousticness);
        let is_z_polarized = at_least(0.9, &z_polarization);
        let is_xy_polarized = at_least(0.9, &xy_polarization);
        let is_layer_acoustic = and(&at_least(0.95, &layer_acousticness), &not(&is_acoustic));

        let is_shear = and(&is_layer_acoustic, &is_xy_polarized);
        let is_layer_breathing = and(&is_layer_acoustic, &is_z_polarized);

        let mut out = vec![];

        if let Some(freqs) = select(&is_acoustic, &frequency) {
            let value = ::serde_yaml::to_value(freqs).unwrap();
            out.push(make_nested_mapping(&["acoustic"], value));
        }

        if let Some(freqs) = select(&is_shear, &frequency) {
            let value = ::serde_yaml::to_value(freqs).unwrap();
            out.push(make_nested_mapping(&["shear"], value));
        }

        if let Some(freqs) = select(&is_layer_breathing, &frequency) {
            let value = ::serde_yaml::to_value(freqs).unwrap();
            out.push(make_nested_mapping(&["layer-breathing"], value));
        }

        // For gamma probs, don't bother with all layers; just a couple.
        [0, 1].iter().for_each(|&layer_n| {
            let probs = ev_layer_gamma_probs.as_maybe().map(|d| d.0[layer_n].to_vec());
            let layer_key = format!("layer-{}", layer_n + 1);

            let pred = at_least(settings.layer_gamma_threshold, &probs);
            if let Some(tuples) = select(&pred, &enumerate(&zip(&frequency, &probs))) {
                #[derive(Serialize)]
                struct Item {
                    index: usize,
                    frequency: f64,
                    probability: f64,
                }

                let items = tuples.into_iter().map(|(index, &(frequency, probability))| {
                    Item { index, frequency, probability }
                }).collect_vec();

                let value = ::serde_yaml::to_value(items).unwrap();
                out.push(make_nested_mapping(&["layer-gammas", &layer_key], value));
            }
        });

        match out.len() {
            0 => no_summary(),
            _ => {
                let yaml = out.into_iter().fold(no_summary(), merge_summaries);
                make_nested_mapping(&["modes", "gamma"], yaml)
            }
        }
    }
}

// Color range used by most columns that contain probabilities in [0, 1]
fn default_prob_color_range() -> ColorByRange<f64> {
    use ::ansi_term::Colour::{Red, Cyan, Yellow, Black};
    ColorByRange::new(vec![
        (0.999, Cyan.bold()),
        (0.9,   Cyan.normal()),
        (0.1,   Yellow.normal()),
        (1e-4,  Red.bold()),
        (1e-10, Red.normal()),
    ],          Black.normal()) // make zeros "disappear"
}

/// Simple Display impl for probabilities (i.e. from 0 to 1).
///
/// Shows a float at dynamically-chosen fixed precision.
#[derive(Debug, Copy, Clone)]
pub struct FixedProb(f64, usize);
impl fmt::Display for FixedProb {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    { write!(f, "{:width$.prec$}", self.0, prec = self.1, width = self.1 + 2) }
}

/// Specialized display impl for probabilities (i.e. from 0 to 1)
/// which may be extremely close to either 0 or 1.
///
/// This should only be used on a value which is computed from a sum of
/// non-negative values; if it were computed from a sum where cancellation may
/// occur, then magnitudes close to 0 or 1 would be too noisy to be meaningful.
///
/// Always displays with a width of 7 characters.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct DisplayProb(f64);
impl fmt::Display for DisplayProb {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let log10_1p = |x: f64| x.ln_1p() / ::std::f64::consts::LN_10;

        // NOTE: This used to deliberately reject values precisely equal to zero,
        //       but I could not recall why, so I loosened the restriction.
        //       (that said, the assertion never failed when it did reject zeros...)
        //
        // NOTE: It still rejects slightly negative values ("numerical zero")
        //       because it should not be used on sums where cancellation may occur.
        assert!(
            0.0 <= self.0 && self.0 < 1.0 + 1e-5,
            "bad probability: {}", self.0);

        if self.0 >= 1.0 {
            write!(f, "{:>7}", 1.0)
        } else if self.0 == 0.0 {
            write!(f, "{:>7}", 0.0) // don't do log of 0
        } else if self.0 - 1e-3 <= 0.0 {
            write!(f, "  1e{:03}", self.0.log10().round())
        } else if self.0 + 1e-3 >= 1.0 {
            write!(f, "1-1e{:03}", log10_1p(-self.0).round())
        } else {
            write!(f, "{:<7.5}", self.0)
        }
    }
}

use self::columns::{Columns, Color, Precision, fixed_prob_column, display_prob_column};
mod columns {
    use super::*;
    use std::iter::{Chain, Once, once};
    use self::inductive::hlist::HList1;

    #[derive(Debug, Clone)]
    pub struct Columns<T = String> {
        pub header: String,
        pub entries: Vec<T>,
    }

    impl<T: fmt::Display> IntoIterator for Columns<T> {
        type IntoIter = Chain<Once<String>, ::std::vec::IntoIter<String>>;
        type Item = String;

        fn into_iter(self) -> Self::IntoIter
        {
            // HACK: so many unnecessary allocations...
            let entries = self.entries.into_iter().map(|x| x.to_string()).collect_vec();
            once(self.header).chain(entries)
        }
    }

    /// Join columns side-by-side.
    pub fn join(columns: &[Columns], sep: &str) -> Columns {
        let mut columns = columns.iter();
        let mut out = columns.next().expect("can't join 0 columns").clone();
        for column in columns {
            out.header = out.header + sep + &column.header;
            for (dest, src) in zip_eq(&mut out.entries, &column.entries) {
                *dest += sep;
                *dest += src;
            }
        }
        out
    }

    /// Factors out logic common between the majority of columns.
    ///
    /// Values are string-formatted by some mapping function, and optionally colorized
    /// according to the magnitude of their value.
    fn quick_column<C, D, F>(painter: &PaintAs<D, C>, header: &str, values: &[C], width: usize, mut show: F) -> Columns
        where
            C: PartialOrd,
            F: StdFnMut(&C) -> D,
            D: fmt::Display,
    { Columns {
        header: format!(" {:^width$}", header, width = width),
        entries: {
            values.iter()
                .map(|x| format!(" {}", painter.paint_as(x, show(x))))
                .collect()
        },
    }}

    pub struct Precision(pub usize);
    pub enum Color {
        Colorful,
        Colorless,
    }

    pub fn fixed_prob_column(color: Color, precision: Precision, header: &str, values: &[f64]) -> Columns
    {
        let painter: Box<PaintAs<_, f64>> = match color {
            Color::Colorful  => Box::new(default_prob_color_range()),
            Color::Colorless => Box::new(NullPainter),
        };
        quick_column(&*painter, header, values, precision.0 + 2, |&x: &_| FixedProb(x, precision.0))
    }

    pub fn display_prob_column(color: Color, header: &str, values: &[f64]) -> Columns
    {
        let painter: Box<PaintAs<_, f64>> = match color {
            Color::Colorful  => Box::new(default_prob_color_range()),
            Color::Colorless => Box::new(NullPainter),
        };
        quick_column(&*painter, header, values, 7, |&x: &_| DisplayProb(x))
    }
}
