
extern crate serde;
#[macro_use] extern crate serde_derive;
#[cfg_attr(test, macro_use)] extern crate serde_json;

#[macro_use]
extern crate sp2_assert_close;
extern crate sp2_array_utils;
extern crate sp2_slice_math;

// because if you're writing a json literal,
// then you probably know it's valid...
#[cfg(test)]
macro_rules! from_json {
    ($($arg:tt)*)
    => { ::serde_json::from_value(json!($($arg)*)).unwrap() };
}

#[macro_use]
extern crate log;
#[macro_use]
extern crate itertools;
extern crate rand;
extern crate ordered_float;
#[cfg(test)]
extern crate env_logger;

#[cfg(test)]
pub(crate) mod test_functions;
pub(crate) mod util;
pub(crate) mod stop_condition;
pub mod acgsd;
pub(crate) mod linesearch;
pub(crate) mod new_linesearch;
pub use acgsd::acgsd;
pub use new_linesearch::linesearch;
pub(crate) mod reporting;