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

pub type IoError = ::std::io::Error;
pub type YamlError = ::serde_yaml::Error;
pub type Shareable = Send + Sync + 'static;

pub use self::filetypes::{disp_yaml, force_sets, symmetry_yaml};

pub use self::filebased::{DirWithDisps, DirWithForces, DirWithBands};
pub use self::filebased::BandsBuilder;
pub use self::filebased::Builder;

mod filetypes;
mod npy;
mod filebased;
mod fs_util;

pub use errors::*;
pub(crate) mod errors {
    pub type IoResult<T> = ::std::io::Result<T>;
    error_chain!{
        links {
            Structure(::rsp2_structure::Error, ::rsp2_structure::ErrorKind);
            StructureIo(::rsp2_structure_io::Error, ::rsp2_structure_io::ErrorKind);
        }

        foreign_links {
            Io(::std::io::Error);
            Json(::serde_json::Error);
            Yaml(::serde_yaml::Error);
        }

        errors {
            PhonopyFailed(status: ::std::process::ExitStatus) {
                description("phonopy exited unsuccessfully"),
                display("phonopy exited unsuccessfully ({})", status),
            }
            NonPrimitiveStructure {
                description("attempted to compute symmetry of a supercell"),
                display("attempted to compute symmetry of a supercell"),
            }
        }
    }
}

/// This module only exists to have its name appear in logs.
/// It marks phonopy's stdout.
mod stdout {
    pub fn log(s: &str)
    { info!("{}", s) }
}

/// This module only exists to have its name appear in logs.
/// It marks phonopy's stderr.
mod stderr {
    pub fn log(s: &str)
    { warn!("{}", s) }
}

pub trait As3<T> {
    fn as_3(&self) -> (&T, &T, &T);
}

impl<T> As3<T> for [T; 3] {
    fn as_3(&self) -> (&T, &T, &T)
    { (&self[0], &self[1], &self[2]) }
}

impl<T> As3<T> for (T, T, T) {
    fn as_3(&self) -> (&T, &T, &T)
    { (&self.0, &self.1, &self.2) }
}
