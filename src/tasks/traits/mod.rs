pub use ::std::result::Result as StdResult;

#[macro_use]
mod macros;
mod pathlike;
mod source;
mod util;
pub(crate) mod save;

pub mod alternate; // Fn traits
pub use self::pathlike::{AsPath, HasTempDir};
pub use self::save::{Save, Load};
pub use self::source::*;
pub use self::util::IsNewtype;
