pub extern crate clap;

// * clap_app! is full of problems:
//   - --nice versus --("ew-gross")
//   - no shorthand for takes_value?! (or value_name, either)
//   - can't set e.g. required(true) on a @group?
//     (or maybe you can, but honestly, who could even tell!?)
//   - it only builds the whole entire App, so when you do run into its
//     limitations there's nothing you can do (except stop using clap_app).
//     A family of macros for different expressions is much better.
// * We want to ensure there's no `arg.multiple(true).takes_value(true)`
//     without also `arg.number_of_values(1)`

/// Alternative to `clap_app!`.
///
/// Produces a `clap::Arg`.
///
/// See bottom of source file for examples.
///
/// In the future this may be just one member of
/// a larger family of macros for producing `App`s and `Group`s, etc.
#[macro_export]
macro_rules! arg {
    ($($token:tt)+)
    => {arg_impl!{ #start# $($token)+ }};
}

#[macro_export]
macro_rules! arg_impl {
    (#start# $($rest:tt)*)
    => {arg_impl!{ #required# $($rest)* }};

    //--- error helpers ---
    (!expected! ($expected:expr) $first:tt $($rest:tt)*)
    => { compile_error!{concat!("expected ", $expected, ", found '", stringify!(first), "'")} };
    (!expected! ($expected:expr))
    => { compile_error!{concat!("expected ", $expected, ", found EOF")} };

    //--- #required# ---
    // required status and name
    // * name  - required
    // ? name  - optional
    //   name  - infer based on option/positional

    (#required# $name:ident $uhoh:ident $($rest:tt)*)
    => {
        compile_error!{concat!{
            "found two consecutive idents, did you mean to write '",
            stringify!($name), "=", stringify!($uhoh), "'?"
        }}
    };

    (#required# * $name:ident $($rest:tt)*)
    => {arg_impl!{ [(*)($name)] #construct# $($rest)* }};

    (#required# ? $name:ident $($rest:tt)*)
    => {arg_impl!{ [(?)($name)] #construct# $($rest)* }};

    (#required#   $name:ident $($rest:tt)*)
    => {arg_impl!{ [()($name)] #construct# $($rest)* }};

    (#required# $($kaboom:tt)*)
    => {arg_impl!{ !expected! ("'*', '?', or ident") $($kaboom)* }};

    //--- #construct# ---
    // Handle the name.
    // ('kind' is saved until later, when we know whether it is positional)
    (
        [$kind:tt ($name:ident)]
        #construct# $($rest:tt)*
    )
    => {arg_impl!{
        ($crate::clap::Arg::with_name(stringify!($name)))
        [$kind ()] // () initializes an option list for #opts#
        #opts# $($rest)*
    }};

    //--- #opts# ---
    // [-L][--long-form][--even-more-names]
    (
        // Munch the options one at a time into the $opt list.
        ($b:expr) [$kind:tt ($($opt:tt)*)]
        #opts# [$($opt_tok:tt)+] $($rest:tt)* // find a [] tt
    )
    => {arg_impl!{
        ($b) [$kind ($($opt)* [$($opt_tok)+] )] // append to end
        #opts# $($rest)* // check for another
    }};

    (
        ($b:expr) [$kind:tt ($($opt:tt)*)]
        #opts# $($rest:tt)*  // end of [] tts
    )
    => {arg_impl!{
        ($b) [$kind ($($opt)*)]
        #handle-kind# $($rest)*
    }};

    //--- #handle-kind# ---
    // Sets the required() flag
    (
        ($b:expr) [$kind:tt $opts:tt]
        #handle-kind# $($rest:tt)*
    )
    => {arg_impl!{
        ($b.required(arg_impl!(&required-flag& $kind $opts)))
        [$kind $opts]
        #handle-opts# $($rest)*
    }};

    //--- #handle-opts# ---
    // Does .short() and .long().
    (
        ($b:expr) [$kind:tt ($($opt:tt)*)]
        #handle-opts# $($rest:tt)*
    )
    => {arg_impl!{
        ({
            let b = $b;
            $( let b = arg_impl!(&add-opt& (b) $opt); )*
            b
        }) []
        #value# $($rest)*
    }};

    //--- #value# ---
    // =FILE
    // * For an option, this makes it take an option argument.
    // * For a positional, this names the meta argument.
    (
        ($b:expr) []
        #value# =$NAME:ident $($rest:tt)*
    )
    => {arg_impl!{
        ({
            // (this same code is run for both options and positionals;
            //  thankfully, takes_value(true) on a positional is harmless)
            // (it also appears that this is the case for number_of_values(1) as well.
            //  Something which, by the way, IS bad when paired with takes_value(false). )
            $b.takes_value(true).value_name(stringify!($NAME)).number_of_values(1)
        })
        // We'll need to remember this for a future decision...
        []
        #multiple# $($rest)*
    }};

    (
        // No =NAME
        ($b:expr) []
        #value# $($rest:tt)*
    )
    => {arg_impl!{
        ($b) []
        #multiple# $($rest)*
    }};

    //--- #multiple# ---
    // ...
    (
        ($b:expr) []
        #multiple# ... $($rest:tt)*
    )
    => {arg_impl!{
        ($b.multiple(true)) []
        #help# $($rest)*
    }};

    (
        ($b:expr) []
        #multiple# $($rest:tt)*
    )
    => {arg_impl!{
        ($b) []
        #help# $($rest)*
    }};

    //--- #help# ---
    // "help message"
    (($b:expr) [] #help#)
    => { $b };
    (($b:expr) [] #help# $help:expr)
    => { $b.help($help) };
    (($($kaboom:tt)*) [] #help#)
    => {
        // (so many optional pieces of syntax have gone by that it's
        //  hard to say what should have been here)
        arg_impl!{ !expected! ("something valid") $($kaboom)* }
    };

    //------------------------
    // Helpers that expand to `expr`
    (&required-flag& (*) $t:tt) => { true };
    (&required-flag& (?) $t:tt) => { false };
    (&required-flag& () (/*positional*/)) => { true };
    (&required-flag& () ( $($opts:tt)+ )) => { false };

    (&add-opt& ($b:expr) [--$($opt:tt)*])
    => { $b.long( concat!($(stringify!($opt)),*) ) };
    (&add-opt& ($b:expr) [-$($opt:tt)*])
    => { $b.short( concat!($(stringify!($opt)),*) ) };
}

#[test]
fn examples() {
    let _ = arg!(*output [-o][--output]=OUTDIR "output directory");
    let _ = arg!(*config [-c][--config]=CONFIG "settings yaml");
    let _ = arg!( input=POSCAR "POSCAR");
    let _ = arg!( force [-f][--force] "replace existing output directories");
    let _ = arg!( save_bands [--save-bands] "save phonopy directory with bands at gamma");

    // not covered above:    -c file1 -c file2
    let _ = arg!( config [-c][--config]=CONFIG... "POSCAR");

    // not covered above:  an optional positional
    let _ = arg!(?input=INPUT "Input file. Defaults to stdin.");
}
