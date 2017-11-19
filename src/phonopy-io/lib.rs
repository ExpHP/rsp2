extern crate rsp2_kets;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_byte_tools_plus_float as byte_tools;
extern crate slice_of_array;

#[macro_use] extern crate error_chain;
#[macro_use] extern crate nom;
#[macro_use] extern crate log;
#[macro_use] extern crate serde_derive;
extern crate serde_yaml;
extern crate serde_json;
extern crate rsp2_tempdir as tempdir;
extern crate rsp2_fs_util;

pub type IoError = ::std::io::Error;
pub type YamlError = ::serde_yaml::Error;
pub type Shareable = Send + Sync + 'static;

pub use self::filetypes::{conf, disp_yaml, force_sets, symmetry_yaml};

mod filetypes;
pub mod npy;

pub use errors::*;
pub(crate) mod errors {
    use ::std::path::PathBuf;

    pub type IoResult<T> = ::std::io::Result<T>;
    error_chain!{
        links {
            Fs(::rsp2_fs_util::Error, ::rsp2_fs_util::ErrorKind);
            Structure(::rsp2_structure::Error, ::rsp2_structure::ErrorKind);
            StructureIo(::rsp2_structure_io::Error, ::rsp2_structure_io::ErrorKind);
        }

        foreign_links {
            Io(::std::io::Error);
            Json(::serde_json::Error);
            Yaml(::serde_yaml::Error);
        }

        errors {
        }
    }
}
