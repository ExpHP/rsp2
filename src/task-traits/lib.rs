extern crate serde;
extern crate serde_yaml;
extern crate serde_json;
#[macro_use]
extern crate error_chain;

// extern crate failure;
// #[macro_use]
// extern crate failure_derive;

error_chain!{
    links {
        Fs(::rsp2_fs_util::Error, ::rsp2_fs_util::ErrorKind);
    }

    foreign_links {
        Io(::std::io::Error);
        Json(::serde_json::Error);
        Yaml(::serde_yaml::Error);
    }
}

extern crate rsp2_tempdir;
extern crate rsp2_fs_util;

pub use ::std::result::Result as StdResult;

#[macro_use]
mod macros;
mod pathlike;
mod save;
mod source;
mod util;

pub mod alternate; // Fn traits
pub use pathlike::{AsPath, HasTempDir};
pub use save::{Save, Load};
pub use source::*;
pub use util::IsNewtype;
