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
extern crate rsp2_tempdir as tempdir;

pub type IoError = ::std::io::Error;
pub type YamlError = ::serde_yaml::Error;
pub type Shareable = Send + Sync + 'static;

pub(crate) use self::filetypes::{Displacements, DispYaml};
pub use self::filetypes::{disp_yaml, force_sets};

pub mod cmd;
mod filetypes;
mod npy;

error_chain!{
    foreign_links {
        Io(::std::io::Error);
        Yaml(::serde_yaml::Error);
    }

    errors {
        PhonopyExitCode(code: u32) {
            description("phonopy exited unsuccessfully"),
            display("phonopy exited unsuccessfully ({})", code),
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
