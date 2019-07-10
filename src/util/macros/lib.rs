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

/// Simpler than the `collect!` macro from grabbag_macros,
/// and, more importantly, supports a terminal comma.
///
/// It has the obvious definition in terms of `vec![]` and collect.
#[macro_export]
macro_rules! collect {
    ($($e:tt)*) => { vec![$($e)*].into_iter().collect() };
}

/// Does `::serde_json::from _value(json!($($arg)*)).unwrap()`
///
/// Why? Because if you're writing a json literal, then you're probably
/// already quite certain that it is valid!
#[macro_export]
macro_rules! from_json {
    ($($arg:tt)*) => { serde_json::from_value(serde_json::json!($($arg)*)).unwrap() };
}

#[macro_export]
macro_rules! zip_eq {
    // (I have a variadic, token munching version lying around somewhere, but frankly
    //  if you need more than four you should just use indices)
    ($a:expr $(,)*) => {
        $a.into_iter().map(|a| (a,))
    };
    ($a:expr, $b:expr $(,)*) => {
        $crate::zip_eq($a, $b)
    };
    ($a:expr, $b:expr, $c:expr $(,)*) => {
        $crate::zip_eq($crate::zip_eq($a, $b), $c)
            .map(|((a, b), c)| (a, b, c))
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)*) => {
        $crate::zip_eq($crate::zip_eq($crate::zip_eq($a, $b), $c), $d)
            .map(|(((a, b), c), d)| (a, b, c, d))
    };
}

pub fn zip_eq<As, Bs>(a: As, b: Bs) -> std::iter::Zip<As::IntoIter, Bs::IntoIter>
where
    As: IntoIterator, As::IntoIter: ExactSizeIterator,
    Bs: IntoIterator, Bs::IntoIter: ExactSizeIterator,
{
    let (a, b) = (a.into_iter(), b.into_iter());
    assert_eq!(a.len(), b.len());
    a.zip(b)
}

// do something only the first time the macro is encountered
#[macro_export]
macro_rules! once {
    ($($tok:tt)*) => {{
        use std::sync::{Once, ONCE_INIT};
        static ONCE: Once = ONCE_INIT;
        ONCE.call_once(|| { $($tok)* });
    }};
}

#[macro_export]
macro_rules! _log_once_impl {
    ($mac:ident!($($arg:tt)*)) => {
        once!{
            // Explicitly label one-time messages to discourage reasoning
            // along the lines of "well it didn't say anything THIS time"
            $mac!("{} (this message will not be shown again)", format_args!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! warn_once { ($($arg:tt)*) => { _log_once_impl!{warn!($($arg)*)} }; }
#[macro_export]
macro_rules! info_once { ($($arg:tt)*) => { _log_once_impl!{info!($($arg)*)} }; }
#[macro_export]
macro_rules! trace_once { ($($arg:tt)*) => { _log_once_impl!{trace!($($arg)*)} }; }
#[macro_export]
macro_rules! debug_once { ($($arg:tt)*) => { _log_once_impl!{debug!($($arg)*)} }; }

#[macro_export]
macro_rules! named_block {
    ($lt:lifetime: $block:block) => { $lt: loop { break $block } };
}

/// Generates the type [[...[T; nN]; ...; n1]; n0].
#[macro_export]
macro_rules! nd {
    ($T:ty; $n0:expr $(;$n:expr)*)
    => { [nd!($T $(;$n)*); $n0] };

    ($T:ty)
    => { $T };
}

#[macro_export]
macro_rules! matches {
    ($pat:pat, $place:expr) => {
        match $place {
            $pat => true,
            _ => false,
        }
    };
}
