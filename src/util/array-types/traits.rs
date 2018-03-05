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
    impl_field!({$T:ty}) => {
        impl Field for $T { }
        impl field::Sealed for $T { }
    };
}

gen_each!{
    @{ring}
    impl_ring!({$T:ty}) => {
        impl Ring for $T { }
        impl ring::Sealed for $T { }
    };
}

gen_each!{
    @{semiring}
    impl_semiring!({$T:ty}) => {
        impl Semiring for $T { }
        impl semiring::Sealed for $T { }
    };
}

/// Internal-use marker traits for generic implementations.
///
/// By using these traits instead of generating separate impls for
/// every element type, we can improve compilation time through
/// lazier codegen.
///
/// Some of these traits are 'unsafe' even though they cannot be
/// named by the consumer.  This is for the benefit of the mantainer
/// of this crate.
pub(crate) mod internal {
    use ::std::ops::{Add, Sub, Mul, Div, Neg};

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

    /// # Safety
    ///
    /// Unsafe code in this crate may make a variety of assumptions
    /// about the semantics of functionality made available through
    /// this trait (for instance, with respect to panic behavior).
    /// Adding a new impl will require review of such code.
    pub unsafe trait PrimitiveSemiring
        : Sized + Copy + Clone + Default
        + PartialEq + PartialOrd
        + SelfAdd + RefAdd
        + SelfMul + RefMul
        + ::std::iter::Sum
        + ::std::iter::Product
    {
        fn from_uint(u: u8) -> Self;
        #[inline(always)] fn zero() -> Self { Self::from_uint(0) }
        #[inline(always)] fn one() -> Self { Self::from_uint(1) }
        #[inline(always)] fn two() -> Self { Self::from_uint(2) }
    }

    gen_each!{
        @{semiring}
        impl_primitive_semiring!({$T:ty})
        => {
            unsafe impl PrimitiveSemiring for $T {
                #[inline(always)] fn from_uint(u: u8) -> $T { u as $T }
            }
        };
    }

    /// # Safety
    ///
    /// Unsafe code in this crate may make a variety of assumptions
    /// about the semantics of functionality made available through
    /// this trait (for instance, with respect to panic behavior).
    /// Adding a new impl will require review of such code.
    pub unsafe trait PrimitiveRing
        : PrimitiveSemiring
        + SelfSub + RefSub + SelfNeg
    {
        fn from_int(i: i8) -> Self;
    }

    gen_each!{
        @{ring}
        impl_primitive_ring!({$T:ty})
        => {
            unsafe impl PrimitiveRing for $T {
                #[inline(always)] fn from_int(i: i8) -> $T { i as $T }
            }
        };
    }

    /// # Safety
    ///
    /// Unsafe code in this crate may make a variety of assumptions
    /// about the semantics of functionality made available through
    /// this trait (for instance, with respect to panic behavior).
    /// Adding a new impl will require review of such code.
    pub unsafe trait PrimitiveFloat
        : PrimitiveRing
        + SelfDiv + RefDiv
    {
        fn sqrt(self) -> Self;
    }

    gen_each!{
        @{float}
        impl_primitive_float!({$T:ty})
        => {
            unsafe impl PrimitiveFloat for $T {
                #[inline(always)] fn sqrt(self) -> $T { self.sqrt() }
            }
        };
    }
}