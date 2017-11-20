extern crate rsp2_kets;
extern crate rsp2_structure;
extern crate rsp2_byte_tools_plus_float as byte_tools;

#[macro_use] extern crate error_chain;
#[macro_use] extern crate nom;
#[macro_use] extern crate serde_derive;
extern crate serde_yaml;

pub type IoError = ::std::io::Error;
pub type YamlError = ::serde_yaml::Error;

pub use self::filetypes::{conf, disp_yaml, force_sets, symmetry_yaml};
pub use self::filetypes::{Conf};
pub use self::filetypes::symmetry_yaml::SymmetryYaml;
pub use self::filetypes::disp_yaml::DispYaml;
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
        }

        errors {}
    }
}
