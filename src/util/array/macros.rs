/// Generates the type [[...[T; nN]; ...; n1]; n0].
macro_rules! nd {
    ($T:ty; $n0:expr $(;$n:expr)*)
    => { [nd!($T $(;$n)*); $n0] };

    ($T:ty)
    => { $T };
}

// Brother!{ItsBeen, TooLong}
macro_rules! Brother {
    ($Array: ty, $E: ty)
    => { <$Array as WithElement<$E>>::Type };
}

macro_rules! Element {
    ($Arr: ty)
    => { <$Arr as IsArray>::Element };
}

macro_rules! MatrixElement {
    ($Mat: ty)
    => { Element![Element![$Mat]] };
}

/// Higher-order macro that iterates over a cartesian product.
///
/// Useful for generating impls involving opaque traits of the sort
/// sometimes seen used as bounds in public APIs.
///
/// It takes a number of groups of token trees and a suitable definition
/// for a callback macro, and it calls the macro with one token tree from
/// each group in order.
///
/// See the examples module in the source for example usage.
macro_rules! cartesian {
    (
        $([$($groups:tt)*])*
        $mac:ident!($($mac_match:tt)*)
        => {$($mac_body:tt)*}$(;)*
    )
    => {
        macro_rules! $mac {
            ($($mac_match)*) => {$($mac_body)*};
        }
        cartesian__!{ @product::next($([$($groups)*])*) -> ($mac!()) }
    };
}

/// implementation detail, go away
macro_rules! cartesian__ {

    (@product::next([$($token:tt)+] $($rest:tt)*) -> $cb:tt)
    => { cartesian__!{ @product::unpack([$($token)+] $($rest)*) -> $cb } };
    // base case; direct product of no arguments
    (@product::next() -> ($mac:ident!($($args:tt)*)))
    => {$mac!{$($args)*}};

    // Each direct product in the invocation incurs a fixed number of recursions
    //  as we replicate the macro.  First, we must smash anything we want to replicate
    //  into a single tt that can be matched without repetitions.  Do this to `rest`.
    (@product::unpack([$($token:tt)*] $($rest:tt)*) -> $cb:tt)
    => {cartesian__!{ @product::unpack_2([$($token)*] [$($rest)*]) -> $cb }};

    // Replicate macro for each token.
    (@product::unpack_2([$($token:tt)*] $rest:tt) -> $cb:tt)
    => { $( cartesian__!{ @product::unpack_3($token $rest) -> $cb } )* };

    // Expand the unparsed arguments back to normal;
    // add the token into the macro call
    (@product::unpack_3($token:tt [$($rest:tt)*]) -> ($mac:ident!($($args:tt)*)))
    => {cartesian__!{ @product::next($($rest)*) -> ($mac!($($args)*$token)) }};
}

/// `cartesian!` with some predefined groups used in the library's public API.
///
/// See the examples module in the source for example usage.
macro_rules! gen_each {
    ($($arg:tt)*) => { gen_each__!{[$($arg)*] -> []} };
}

macro_rules! gen_each__ {
    //----------------------------
    // Groups using the standard syntax supported by cartesian

    ([[$($alternatives:tt)*] $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        $($alternatives)*
    ]] }};

    //----------------------------
    // Special groups of the form @{...}

    // NOTE: This macro is what truly defines the set of members
    //       for each opaque trait.

    // Types that implement Field
    ([@{field} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {f32} {f64}
    ]] }};

    // Types that implement Ring
    ([@{ring} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {f32} {f64}
        {i8} {i16} {i32} {i64} {isize}
    ]] }};

    // Types that implement Semiring
    ([@{semiring} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {f32} {f64}
        {i8} {i16} {i32} {i64} {isize}
        {u8} {u16} {u32} {u64} {usize}
    ]] }};

    // PrimitiveFloat types
    ([@{float} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {f32} {f64}
    ]] }};

    // Fixed sized vector types
    ([@{Vn} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {V2} {V3} {V4}
    ]] }};

    // ...along with their size
    ([@{Vn_n} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {V2 2} {V3 3} {V4 4}
    ]] }};

    // Matrices of generic width, and their number of rows
    ([@{Mn_n} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {M2 2} {M3 3} {M4 4}
    ]] }};

    // Square matrices, their vector types, and size
    ([@{Mnn_Mn_Vn_n} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        {M22 M2 V2 2} {M33 M3 V3 3} {M44 M4 V4 4}
    ]] }};

    ([@{0...4} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        { 0} { 1} { 2} { 3} { 4}
    ]] }};

    ([@{1...4} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
             { 1} { 2} { 3} { 4}
    ]] }};

    ([@{0...8} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        { 0} { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8}
    ]] }};

    ([@{1...8} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
             { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8}
    ]] }};

    ([@{0...16} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        { 0} { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8} { 9}
        {10} {11} {12} {13} {14} {15} {16}
    ]] }};

    ([@{1...16} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
             { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8} { 9}
        {10} {11} {12} {13} {14} {15} {16}
    ]] }};

    ([@{0...32} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        { 0} { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8} { 9}
        {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}
        {20} {21} {22} {23} {24} {25} {26} {27} {28} {29}
        {30} {31} {32}
    ]] }};

    ([@{1...32} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
             { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8} { 9}
        {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}
        {20} {21} {22} {23} {24} {25} {26} {27} {28} {29}
        {30} {31} {32}
    ]] }};

    ([@{0...64} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
        { 0} { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8} { 9}
        {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}
        {20} {21} {22} {23} {24} {25} {26} {27} {28} {29}
        {30} {31} {32} {33} {34} {35} {36} {37} {38} {39}
        {40} {41} {42} {43} {44} {45} {46} {47} {48} {49}
        {50} {51} {52} {53} {54} {55} {56} {57} {58} {59}
        {60} {61} {62} {63} {64}
    ]] }};

    ([@{1...64} $($rest:tt)*] -> [$($done:tt)*])
    => { gen_each__!{[$($rest)*] -> [$($done)* [
             { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8} { 9}
        {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}
        {20} {21} {22} {23} {24} {25} {26} {27} {28} {29}
        {30} {31} {32} {33} {34} {35} {36} {37} {38} {39}
        {40} {41} {42} {43} {44} {45} {46} {47} {48} {49}
        {50} {51} {52} {53} {54} {55} {56} {57} {58} {59}
        {60} {61} {62} {63} {64}
    ]] }};

    // Finally: Delegate to `cartesian`
    ([$mac:ident!$($defn_args:tt)*] -> [$($groups:tt)*])
    => {
        cartesian!{
            $($groups)*
            $mac!$($defn_args)*
        }
    };
}

/// Synthesize a `Vn` vector type from a `tt` of the size.
macro_rules! V {
    (2, $X:ty) => { V2<$X> };
    (3, $X:ty) => { V3<$X> };
    (4, $X:ty) => { V4<$X> };
}

/// Synthesize an `Mn` matrix type from a `tt` of the size.
macro_rules! M {
    (2, $V:ty) => { M2<$V> };
    (3, $V:ty) => { M3<$V> };
    (4, $V:ty) => { M4<$V> };
}

#[cfg(test)]
mod examples {
    mod cartesian {
        trait Trait { }
        // NOTE: Braces around the alternatives are not strictly necessary
        //       (cartesian simply iterates over token trees), but they tend
        //       to help group tokens and resolve any would-be ambiguities in
        //       the callback's match pattern.
        cartesian!{
            [{i32} {u32}]
            [{0} {1} {2} {3}]
            unique_name!({$T:ty} {$n:expr})
            => {
                impl Trait for [$T; $n] { }
            }
        }

        #[test]
        fn example_works() {
            fn assert_trait<T:Trait>() {}
            assert_trait::<[u32; 0]>();
            assert_trait::<[i32; 2]>();
        }
    }

    mod gen_each {
        trait Trait { }
        gen_each!{
            @{field}      // equivalent to [{f32} {f64}]
            @{0...4}      // equivalent to [{0} {1} {2} {3} {4}]
            [{i32} {u32}] // explicitly defined groups are still allowed
            unique_name!({$A:ty} {$n:expr} {$B:ty})
            => {
                impl Trait for ([$A; $n], $B) { }
            }
        }

        #[test]
        fn example_works() {
            fn assert_trait<T:Trait>() {}
            assert_trait::<([f64; 3], i32)>();
        }
    }
}
