#[macro_use]
mod macros;
mod pathlike;
pub(crate) mod save;

pub use self::pathlike::{AsPath, HasTempDir};
pub use self::save::{Save, Load};
