
extern crate serde;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate error_chain;
#[macro_use] extern crate serde_json;

#[cfg_attr(test, macro_use)] extern crate sp2_assert_close;
extern crate sp2_array_utils;
extern crate sp2_slice_math;

extern crate either;

// because if you're writing a json literal,
// then you probably know it's valid...
macro_rules! from_json {
    ($($arg:tt)*)
    => { ::serde_json::from_value(json!($($arg)*)).unwrap() };
}

#[macro_use] extern crate log;
#[cfg_attr(test, macro_use)] extern crate itertools;
extern crate rand;
extern crate ordered_float;
#[cfg(test)] extern crate env_logger;

#[cfg(test)] pub(crate) mod test_functions;
pub(crate) mod util;
pub(crate) mod stop_condition;
pub mod acgsd;
pub(crate) mod linesearch;
pub(crate) mod hager_ls;
pub use ::acgsd::acgsd;
pub use ::hager_ls::linesearch;
pub(crate) mod reporting;
