
// NOTE: This is actually the root module of rsp2-array-types.
//       (although you can't put a doc comment here)

pub use self::types::*;
mod types;

pub use self::conv::*;
mod conv;

pub use self::ops::*;
mod ops;

// Expose neatly-named modules, but let the .rs files have names that are close alphabetically.
#[doc(hidden)] pub mod methods_v;
#[doc(hidden)] pub mod methods_m;
pub use self::methods_v as vee;
pub use self::methods_m as mat;

pub use self::methods_v::dot;
