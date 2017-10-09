
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
    ($($arg:tt)*) => { ::serde_json::from_value(json!($($arg)*)).unwrap() };
}
