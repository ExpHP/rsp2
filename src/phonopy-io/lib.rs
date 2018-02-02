extern crate rsp2_kets;
extern crate rsp2_structure;
extern crate rsp2_byte_tools_plus_float as byte_tools;

#[macro_use] extern crate error_chain;
#[macro_use] extern crate nom;
#[macro_use] extern crate serde_derive;
extern crate serde_json;
extern crate serde_yaml;

pub use self::filetypes::{conf, Conf};
pub use self::filetypes::symmetry_yaml::{self, SymmetryYaml};
pub use self::filetypes::disp_yaml::{self, DispYaml};
pub use self::filetypes::force_sets;
pub use self::filetypes::sparse_sets;

mod filetypes;
pub mod npy;

pub use errors::*;
pub(crate) mod errors {
    pub type IoResult<T> = ::std::io::Result<T>;
    error_chain!{
        links {}

        foreign_links {
            Io(::std::io::Error);
            Yaml(::serde_yaml::Error);
            Json(::serde_json::Error);
        }

        errors {}
    }
}
