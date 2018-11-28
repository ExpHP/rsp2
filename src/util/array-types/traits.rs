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

// Traits exposed in public interfaces,
// implemented on finite sets of types rather than more general
//  generic bounds in order to reduce coupling with client crates.

pub use self::semiring::Semiring;
mod semiring {
    /// Trait for scalars with addition and multiplication.
    ///
    /// There are lots and lots and lots and lots and lots of really cool
    /// (and sometimes useful) semiring algebras, like (or, and), and
    /// (xor, and), and (min, plus)/(max, plus)/(max, times),
    /// and square matrices of any semiring scalar type, and...
    ///
    /// But don't get excited. You get primitive floats and integers.
    /// That's all that this API is willing to commit to at the moment.
    /// This trait is sealed to avoid accidental commitments.
    pub trait Semiring : Sealed { }

    pub(super) use self::private::Sealed;
    pub(super) mod private {
        pub trait Sealed { }
    }
}

pub use self::ring::Ring;
mod ring {
    use super::Semiring;

    /// Trait for scalars with addition, multiplication, and subtraction.
    ///
    /// This trait is sealed to avoid accidental commitments.
    /// It doesn't include unsigned integers because a ring must be
    /// closed under negation. (we *could* include `Wrapping`, but bleh)
    pub trait Ring : Semiring + Sealed { }

    pub(super) use self::private::Sealed;
    pub(super) mod private {
        pub trait Sealed { }
    }
}

pub use self::field::Field;
mod field {
    use super::Ring;

    /// Trait for scalars with addition, multiplication, subtraction, and division.
    ///
    /// This trait is sealed to avoid accidental commitments.
    /// It's currently just primitive, real floating point types;
    /// You'll just have to take your rationals and complex numbers elsewhere.
    /// Tragic, I know.
    pub trait Field : Ring + Sealed { }

    pub(super) use self::private::Sealed;
    pub(super) mod private {
        pub trait Sealed { }
    }
}


// Generate the (trivial) impls of Field, Ring, and Semiring.
gen_each!{
    @{field}
    for_each!({$T:ty}) => {
        impl Field for $T { }
        impl field::Sealed for $T { }
    };
}

gen_each!{
    @{ring}
    for_each!({$T:ty}) => {
        impl Ring for $T { }
        impl ring::Sealed for $T { }
    };
}

gen_each!{
    @{semiring}
    for_each!({$T:ty}) => {
        impl Semiring for $T { }
        impl semiring::Sealed for $T { }
    };
}

/// Internal-use marker traits for generic implementations.
///
/// By using these traits instead of generating separate impls for
/// every element type, we can improve compilation time through
/// lazier codegen.
//
// FIXME ummmmm... in hindsight, the compile time claim seems dubious.
//       I wrote that months ago, and I'm not sure what I was thinking then.
//       Does this cause repeated compilation of the same code in
//       downstream crates? :/
pub(crate) mod internal {
    use std::ops::{Add, Sub, Mul, Div, Neg};

    macro_rules! markers {
        ($( $name:ident[$($bound:tt)+]; )+)
        => {$(
            pub trait $name: $($bound)+ { }
            impl<T> $name for T where T: $($bound)+ { }
        )+};
    }

    markers!{
        SelfAdd[Sized + Add<Self, Output=Self>];
        SelfSub[Sized + Sub<Self, Output=Self>];
        SelfMul[Sized + Mul<Self, Output=Self>];
        SelfDiv[Sized + Div<Self, Output=Self>];
        SelfNeg[Sized + Neg<Output=Self>];
        RefAdd[Sized + for<'a> Add<&'a Self, Output=Self>];
        RefSub[Sized + for<'a> Sub<&'a Self, Output=Self>];
        RefMul[Sized + for<'a> Mul<&'a Self, Output=Self>];
        RefDiv[Sized + for<'a> Div<&'a Self, Output=Self>];
    }

    pub trait PrimitiveSemiring
        : Sized + Copy + Clone + Default
        + PartialEq + PartialOrd
        + SelfAdd + RefAdd
        + SelfMul + RefMul
        + num_traits::Zero
        + num_traits::One
        + std::iter::Sum
        + std::iter::Product
    {
        fn from_uint(u: u8) -> Self;
        #[inline(always)] fn two() -> Self { Self::from_uint(2) }
    }

    gen_each!{
        @{semiring}
        for_each!({$T:ty})
        => {
            impl PrimitiveSemiring for $T {
                #[inline(always)] fn from_uint(u: u8) -> $T { u as $T }
            }
        };
    }

    pub trait PrimitiveRing
        : PrimitiveSemiring
        + SelfSub + RefSub + SelfNeg
    {
        fn from_int(i: i8) -> Self;
    }

    gen_each!{
        @{ring}
        for_each!({$T:ty})
        => {
            impl PrimitiveRing for $T {
                #[inline(always)] fn from_int(i: i8) -> $T { i as $T }
            }
        };
    }

    pub trait PrimitiveFloat
        : PrimitiveRing
        + SelfDiv + RefDiv
        + rand::Rand
    {
        // (allow(unused) because these are arbitrarily added as they're needed,
        //  and it's annoying to have to remove them only to possibly later have
        //  to add them back)
        #[allow(unused)] fn sqrt(self) -> Self;
        #[allow(unused)] fn min(self, b: Self) -> Self;
        #[allow(unused)] fn max(self, b: Self) -> Self;
        #[allow(unused)] fn acos(self) -> Self;

        #[allow(unused)] fn uniform_with(rng: impl rand::Rng, _: (Self, Self)) -> Self;
    }

    gen_each!{
        @{float}
        for_each!({$T:ty})
        => {
            impl PrimitiveFloat for $T {
                #[inline(always)] fn sqrt(self) -> $T { self.sqrt() }
                #[inline(always)] fn min(self, b: Self) -> $T { self.min(b) }
                #[inline(always)] fn max(self, b: Self) -> $T { self.max(b) }
                #[inline(always)] fn acos(self) -> $T { self.acos() }

                #[inline(always)] fn uniform_with(mut rng: impl rand::Rng, (lo, hi): (Self, Self)) -> Self {
                    let alpha: Self = rng.gen();
                    lo + (hi - lo) * alpha
                }
            }
        };
    }
}
