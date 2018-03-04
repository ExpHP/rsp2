extern crate rsp2_array_utils;

// HACK: The contents of this crate actually live in rsp2_array_utils because
//       they make use of traits and macros internal to that crate.

// I want these things to be in a separate crate because they have very different
// implications for the code that depends on them; `rsp2_array_utils` is generally
// used like a private dependency, while these types are used like public dependency.
pub use rsp2_array_utils::_rsp2_array_types_impl::*;
