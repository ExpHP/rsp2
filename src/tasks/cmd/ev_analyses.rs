use ::ui::color::{ColorByRange, PaintAs, NullPainter};
use ::ui::cfg_merging::{merge_summaries, make_nested_mapping, no_summary};
use ::util::zip_eq;
use ::types::basis::Basis3;
use ::math::bands::{GammaUnfolder, ScMatrix};
use ::std::rc::Rc;
use ::std::mem;
use ::traits::alternate::{FnOnce};
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

        impl<R, F> Cached for Function<R, F> where F: FnOnce<(), Output=Rc<R>> {
            type Value = R;

            fn call_cached(&self) -> Rc<R> {
                let mut inner = self.0.borrow_mut();
                let result = match mem::replace(&mut *inner, Inner::MidComputation) {
                    Inner::MidComputation => panic!("detected cyclic dependency in Cached objects!"),
                    Inner::Func(f) => f.call_once(()),
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

        impl<W, A, R, F> FnOnce<()> for Closure<A, F>
        where A: Cached<Value=W>, F: FnOnce<Rc<W>, Output=Rc<R>>,
        {
            type Output = Rc<R>;

            fn call_once(self, (): ()) -> Rc<R>
            { self.continuation.call_once(self.arg.call_cached()) }
        }

        impl<W, A, R, F> Cached for Map<A, R, F>
        where A: Cached<Value=W>, F: FnOnce<Rc<W>, Output=Rc<R>>,
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

        impl<D, W, A, R, F> FnOnce<()> for Closure<A, F>
        where A: Cached<Value=W>, F: FnOnce<Rc<W>, Output=D>, D: Cached<Value=R>,
        {
            type Output = Rc<R>;

            fn call_once(self, (): ()) -> Rc<R>
            { self.continuation.call_once(self.arg.call_cached()).call_cached() }
        }

        impl<D, W, A, R, F> Cached for AndThen<A, R, F>
        where A: Cached<Value=W>, F: FnOnce<Rc<W>, Output=D>, D: Cached<Value=R>,
        {
            type Value = R;

            fn call_cached(&self) -> Rc<R>
            { self.0.call_cached() }
        }
    }
}

mod closures {
    use super::*;

    #[derive(Debug, Copy, Clone, Default)]
    pub struct Id;
    impl<X> FnOnce<(X,)> for Id {
        type Output = X;

        fn call_once(self, (x,): (X,)) -> X
        { x }
    }

    #[derive(Debug, Copy, Clone, Default)]
    pub struct FlattenL1;
    impl<A> FnOnce<(A,)> for FlattenL1 {
        type Output = (A,);
        fn call_once(self, (a,): (A,)) -> (A,)
        { (a,) }
    }

    pub type FlattenL2 = Id;

    #[derive(Debug, Copy, Clone, Default)]
    pub struct FlattenL3;
    impl<A, B, C> FnOnce<(((A, B), C),)> for FlattenL3 {
        type Output = (A, B, C);

        fn call_once(self, (((a, b), c),): (((A, B), C),)) -> (A, B, C)
        { (a, b, c) }
    }

    #[derive(Debug, Copy, Clone, Default)]
    pub struct FlattenL4;
    impl<A, B, C, D> FnOnce<((((A, B), C), D),)> for FlattenL4 {
        type Output = (A, B, C, D);

        fn call_once(self, ((((a, b), c), d),): ((((A, B), C), D),)) -> (A, B, C, D)
        { (a, b, c, d) }
    }

    #[derive(Debug, Copy, Clone, Default)]
    pub struct FlattenL5;
    impl<A, B, C, D, E> FnOnce<(((((A, B), C), D), E),)> for FlattenL5 {
        type Output = (A, B, C, D, E);

        fn call_once(self, (((((a, b), c), d), e),): (((((A, B), C), D), E),)) -> (A, B, C, D, E)
        { (a, b, c, d, e) }
    }

    pub trait GetFlattenL {
        type Closure;
    }

    impl GetFlattenL for    ()                { type Closure = FlattenL1; }
    impl GetFlattenL for   ((), ())           { type Closure = FlattenL2; }
    impl GetFlattenL for  (((), ()), ())      { type Closure = FlattenL3; }
    impl GetFlattenL for ((((), ()), ()), ()) { type Closure = FlattenL4; }
}

pub mod option {
    use super::*;
    use ::std::ops::{Deref, DerefMut};
    use ::std::marker::PhantomData;
    #[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Default)]
    pub struct Just<T>(pub T);
    #[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
    pub struct Nothing<T>(pub PhantomData<T>);
    pub type Maybe<T> = Option<T>;

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
    pub type ZipT<A, B> = <A as Zip<B>>::Output;
    pub trait Zip<Rhs: IntoMaybe>: Sized + IntoMaybe {
        type Output: ExpectFrom<Maybe<(Self::TyArg, Rhs::TyArg)>>;

        fn zip(self, other: Rhs) -> Self::Output {
            ExpectFrom::expect_from({
                match (self.into_maybe(), other.into_maybe()) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None,
                }
            })
        }
    }

    pub trait Generic1 { type TyArg1; }
    impl<A> Generic1 for Maybe<A> { type TyArg1 = A; }
    impl<A> Generic1 for Just<A> { type TyArg1 = A; }
    impl<A> Generic1 for Nothing<A> { type TyArg1 = A; }

    #[allow(unused)]
    pub type MapT<A, F> = <A as Map<F>>::Output;
    pub trait Map<F>: Sized {
        type Output;

        fn map(self, f: F) -> Self::Output;
    }

    impl<A, B, F: FnOnce<(A,), Output=B>> Map<F> for Just<A> {
        type Output = Just<B>;

        fn map(self, f: F) -> Self::Output
        { Just(f.call_once((self.0,))) }
    }

    impl<A, B, F: FnOnce<(A,), Output=B>> Map<F> for Maybe<A> {
        type Output = Maybe<B>;

        fn map(self, f: F) -> Self::Output
        { Option::map(self, move |x| f.call_once((x,))) }
    }

    impl<A, B, F: FnOnce<A, Output=B>> Map<F> for Nothing<A> {
        type Output = Nothing<B>;

        fn map(self, _: F) -> Self::Output
        { Default::default() }
    }

    #[allow(unused)]
    pub type ZipNT<Tup> = <() as ZipN<Tup>>::Output;
    pub trait ZipN<Tup> {
        type Output;

        fn zip_n(tup: Tup) -> Self::Output;
    }

    macro_rules! impl_zip_n {
        // special case first iter to correctly handle MegaZipT
        (@next [] [] [] <= [$a0:ident : $A0:ident, $Flatten:path] $($more:tt)*) => {
            impl_zip_n!{@iter [$Flatten] [$A0] [] [[$a0:$A0]] <= $($more)*}
        };
        (@next
            [$MegaZipT:ty] [$($ZipTBounds:tt)*] [$($pairs:tt)*]
            <= [$a0:ident : $A0:ident, $Flatten:path]
               $($more:tt)*
        ) => {
            impl_zip_n!{@iter
                // e.g. [ FlattenL4 ]
                [$Flatten]

                // e.g. [ ZipT<ZipT<ZipT<A, B>, C>, D> ]
                [ZipT<$MegaZipT, $A0>]

                // e.g. [ A: Zip<B>, ZipT<A, B>: Zip<C>, ZipT<ZipT<A, B>, C>: Zip<D> ]
                [$($ZipTBounds)* $MegaZipT: Zip<$A0>,]

                // e.g. [ [a:A] [b:B] [c:C] [d:D] ]
                [$($pairs)* [$a0:$A0]]

                <= $($more)*
            }
        };
        (@next [$MegaZipT:ty] [$($ZipTBounds:tt)*] [$($pairs:tt)*] <= ) => {};

        (@iter $Flatten:tt $MegaZipT:tt $ZipTBounds:tt $pairs:tt <= $($more:tt)*) => {
            impl_zip_n!{@gen $Flatten $MegaZipT $ZipTBounds $pairs}
            impl_zip_n!{@next $MegaZipT $ZipTBounds $pairs <= $($more)*}
        };
        (@gen
            [$Flatten:path]
            [$MegaZipT:ty]
            [$($ZipTBounds:tt)*]
            [[$a0:ident: $A0:ident] $([$a:ident: $A:ident])*]
        ) => {
            // impl<A, B, C, D, E, R> ZipN<(A, B, C, D, E)> for ()
            impl<$A0, $($A,)* R> ZipN<($A0, $($A,)*)> for ()

            // where
            //     A: IntoMaybe,
            //     B: IntoMaybe,
            //     C: IntoMaybe,
            //     D: IntoMaybe,
            //     E: IntoMaybe,
            where
                $A0: IntoMaybe,
                $( $A: IntoMaybe, )*

            //     A: Zip<B>,
            //     ZipT<A, B>: Zip<C>,
            //     ZipT<ZipT<A, B>, C>: Zip<D>,
            //     ZipT<ZipT<ZipT<A, B>, C>, D>: Zip<E>,
            //     ZipT<ZipT<ZipT<ZipT<A, B>, C>, D>, E>: Map<closures::FlattenL5, Output=R>,
                $($ZipTBounds)*
                $MegaZipT: Map<$Flatten, Output=R>,

            // {
            //     type Output = R;
            //
            //     fn zip_n((a, b, c, d, e): (A, B, C, D, E)) -> Self::Output {
            //         a.zip(b).zip(c).zip(d).zip(e).map(closures::FlattenL5)
            //     }
            // }
            {
                type Output = R;

                fn zip_n(($a0, $($a,)*) : ($A0, $($A,)*)) -> Self::Output {
                    $a0 $(.zip($a))* .map($Flatten)
                }
            }
        };

        ($($all:tt)*) => {
            compile_error!{stringify!{$($all)*}}
        };
    }

    impl_zip_n! {
        @next [] [] [] <=
        [a: A, closures::FlattenL1]
        [b: B, closures::Id]
        [c: C, closures::FlattenL3]
        [d: D, closures::FlattenL4]
        [e: E, closures::FlattenL5]
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

    #[allow(unused)]
    pub type AsRefsT<Tup> = <Tup as AsRefs>::Output;
    pub trait AsRefs {
        type Output;

        fn as_refs(self) -> Self::Output;
    }

    macro_rules! impl_as_refs {
        (@eat [$($prev:tt)*], $a0:ident : $A0:ident, $($more:tt)*) => {
            impl_as_refs!{@iter [$($prev)* [$a0:$A0]], $($more)*}
        };
        (@eat [$($prev:tt)*],) => {};

        (@iter $prev:tt, $($more:tt)*) => {
            impl_as_refs!{@gen $prev}
            impl_as_refs!{@eat $prev, $($more)*}
        };
        (@gen [$([$a:ident : $A:ident])+]) => {
            impl<$($A),+> AsRefs for ($($A,)+)
            where
                $( $A: AsRef, )+
            {
                type Output = ($(AsRefT<$A>,)+);

                fn as_refs(self) -> Self::Output {
                    let ($($a,)+) = self;
                    $(
                        let $a = $a.as_ref();
                    )+
                    ($($a,)+)
                }
            }
        };
    }

    impl_as_refs! {
        @eat [],
        a: A, b: B, c: C, d: D,
        e: E, f: F, g: G, h: H,
    }

    pub trait IntoMaybe {
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

    pub trait ExpectFrom<B> {
        fn expect_from(b: B) -> Self;
    }

    impl<A> ExpectFrom<A> for A {
        fn expect_from(a: A) -> Self { a }
    }

    impl<A> ExpectFrom<Maybe<A>> for Just<A> {
        fn expect_from(a: Maybe<A>) -> Self { Just(a.unwrap()) }
    }

    impl<A> ExpectFrom<Maybe<A>> for Nothing<A> {
        fn expect_from(a: Maybe<A>) -> Self
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

    macro_rules! impl_assoc_ty {
        ( $( [$($tpar:ident),*] $Trait:ident [$A:ty] => $Output:ty;)* )
        => {
            $(
                impl<$($tpar),*> $Trait for $A {
                    type Output = $Output;
                }
            )*
        };
        ( $( [$($tpar:ident),*] $Trait:ident [$A:ty, $B:ty] => $Output:ty;)* )
        => {
            $(
                impl<$($tpar),*> $Trait<$B> for $A {
                    type Output = $Output;
                }
            )*
        };
    }

    impl_assoc_ty!{
        // a default impl is used for all
        [A, B] Zip[Nothing<A>, Nothing<B>] => Nothing<(A, B)>;
        [A, B] Zip[  Maybe<A>, Nothing<B>] => Nothing<(A, B)>;
        [A, B] Zip[   Just<A>, Nothing<B>] => Nothing<(A, B)>;
        [A, B] Zip[Nothing<A>,   Maybe<B>] => Nothing<(A, B)>;
        [A, B] Zip[Nothing<A>,    Just<B>] => Nothing<(A, B)>;
        [A, B] Zip[  Maybe<A>,   Maybe<B>] =>   Maybe<(A, B)>;
        [A, B] Zip[  Maybe<A>,    Just<B>] =>   Maybe<(A, B)>;
        [A, B] Zip[   Just<A>,   Maybe<B>] =>   Maybe<(A, B)>;
        [A, B] Zip[   Just<A>,    Just<B>] =>    Just<(A, B)>;
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

    pub trait Use<F> {
        type Output;

        fn go(self, f: F) -> Self::Output;
    }

    impl<'a, F, Z, M, As, Rs, Out> Use<F> for As
    where
        As: option::AsRefs<Output=Rs>,
        (): option::ZipN<Rs, Output=Z>,
        Z: option::Map<F, Output=M>,
        M: option::FoldOk<Output=Out>,
    {
        type Output = Out;

        fn go(self, f: F) -> Self::Output {
            use self::option::ZipN;

            <()>::zip_n(self.as_refs()).map(f).fold_ok()
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
            (&'a InEvEigenvectors,): Use<ev_acousticness::Closure, Output=Result<OutEvAcousticness>>,
            (&'a InEvEigenvectors,): Use<ev_polarization::Closure, Output=Result<OutEvPolarization>>,
            (&'a InAtomLayers, &'a InAtomCoordinates, &'a InLayerScMatrices, &'a InEvEigenvectors,): Use<ev_layer_gamma_probs::Closure, Output=Result<OutEvLayerGammaProbs>>,
            (&'a InAtomLayers, &'a InEvEigenvectors,): Use<ev_layer_acousticness::Closure, Output=Result<OutEvLayerAcousticness>>,

        {ok({
            let Input {
                atom_coords, atom_layers, layer_sc_mats,
                ev_frequencies, ev_eigenvectors,
            } = *self;

            // This is a bit repetitive, perhaps, but the upshot is that
            // it is *impossible* to make a mistake here; it would not compile.

            let ev_acousticness = (ev_eigenvectors,).go(ev_acousticness::Closure)?;

            let ev_polarization = (ev_eigenvectors,).go(ev_polarization::Closure)?;

            let ev_layer_acousticness =
                (atom_layers, ev_eigenvectors,)
                .go(ev_layer_acousticness::Closure)?;

            let ev_layer_gamma_probs =
                (atom_layers, atom_coords, layer_sc_mats, ev_eigenvectors,)
                .go(ev_layer_gamma_probs::Closure)?;

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

                pub struct Closure;
                impl<'a> FnOnce<(($(&'a $Arg,)*),)> for Closure {
                    type Output = Result<$Thing>;

                    fn call_once(self, (($($arg,)*),) : (($(&'a $Arg,)*),) ) -> Self::Output
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

    pub struct EvAcousticness(pub Vec<f64>);
    pub struct Closure;
    impl<'a> FnOnce<((&'a EvEigenvectors,),)> for Closure {
        type Output = Result<EvAcousticness>;

        fn call_once(self, ((ev_eigenvectors,),): ((&EvEigenvectors,),)) -> Self::Output {
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
            where F: FnMut(&A) -> R
        {
            let a = a.as_ref()?;
            Some(a.iter().map(|a| f(a)).collect())
        }

        // (haskell's LiftA2)
        fn map2<B, A, R, F>(a: &Option<Vec<A>>, b: &Option<Vec<B>>, mut f: F) -> Option<Vec<R>>
            where F: FnMut(&A, &B) -> R
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
            F: FnMut(&C) -> D,
            D: fmt::Display,
    { Columns {
        header: format!(" {:^width$}", header, width = width),
        entries: values.iter().map(|x| format!(" {}", painter.paint_as(x, show(x)))).collect(),
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
        quick_column(&*painter, header, values, precision.0 + 2, |&x| FixedProb(x, precision.0))
    }

    pub fn display_prob_column(color: Color, header: &str, values: &[f64]) -> Columns
    {
        let painter: Box<PaintAs<_, f64>> = match color {
            Color::Colorful  => Box::new(default_prob_color_range()),
            Color::Colorless => Box::new(NullPainter),
        };
        quick_column(&*painter, header, values, 7, |&x| DisplayProb(x))
    }
}
