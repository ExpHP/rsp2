/// Generates the type [[...[T; nN]; ...; n1]; n0].
#[macro_export] // useful enough to keep around even if we no longer use it
macro_rules! nd {
    ($T:ty; $n0:expr $(;$n:expr)*)
    => { [nd!($T $(;$n)*); $n0] };

    ($T:ty)
    => { $T };
}

// Brother!{ItsBeen, TooLong}
macro_rules! Brother {
    ($Array:ty, $E:ty)
    => { <$Array as WithElement<$E>>::Type };
}

macro_rules! each_array_size {
    {$mac:ident!{@these [$({$($t:tt)*})*]}} => {
        $( $mac!{ $($t)* } )*
    };
    {$mac:ident!{0...32}} => {
        each_array_size!{$mac!{@these [
            { 0} { 1} { 2} { 3} { 4} { 5} { 6} { 7} { 8} { 9}
            {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}
            {20} {21} {22} {23} {24} {25} {26} {27} {28} {29}
            {30} {31} {32}
        ]}}
    };
}
