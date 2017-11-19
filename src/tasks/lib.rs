// HERE BE DRAGONS

#![recursion_limit="256"] // for error chain...

extern crate rsp2_lammps_wrap;
extern crate rsp2_minimize;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_structure_gen;
extern crate rsp2_phonopy_io;
extern crate rsp2_array_utils;
extern crate rsp2_slice_math;
extern crate rsp2_tempdir;
extern crate rsp2_eigenvector_classify;
extern crate rsp2_fs_util;
#[macro_use] extern crate rsp2_util_macros;

extern crate rayon;
extern crate rand;
extern crate slice_of_array;
extern crate serde;
extern crate ansi_term;
extern crate fern;
extern crate rsp2_kets;
extern crate serde_yaml;
#[macro_use] extern crate serde_json;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate error_chain;

macro_rules! ichain {
    ($e:expr,) => { $e.into_iter() };
    ($e:expr, $($es:expr,)+)
    => { $e.into_iter().chain(ichain!($($es,)+)) };
}

#[macro_use]
mod traits;
mod color;
mod util;
mod config;
mod logging;
mod cmd;
mod integrate_2d;
mod bands;
mod phonopy;
pub use traits::alternate;

pub use ::config::Settings;
pub mod relax_with_eigenvectors {
    pub use ::cmd::run_relax_with_eigenvectors as run;
    pub use ::cmd::CliArgs;
}
pub use ::cmd::run_symmetry_test;
pub use ::cmd::get_energy_surface;
pub use ::cmd::make_force_sets;


pub use ::bands::unfold_phonon;

pub use errors::{Result, ResultExt, Error, ErrorKind, StdResult, IoResult};
mod errors {
    use ::std::path::PathBuf;
    error_chain! {
        foreign_links {
            Io(::std::io::Error);
            Yaml(::serde_yaml::Error);
            Json(::serde_json::Error);
            SetLogger(::log::SetLoggerError);
        }

        links {
            Fsx(::rsp2_fs_util::Error, ::rsp2_fs_util::ErrorKind);
            Structure(::rsp2_structure::Error, ::rsp2_structure::ErrorKind);
            StructureIo(::rsp2_structure_io::Error, ::rsp2_structure_io::ErrorKind);
            StructureGen(::rsp2_structure_gen::Error, ::rsp2_structure_gen::ErrorKind);
            LammpsWrap(::rsp2_lammps_wrap::Error, ::rsp2_lammps_wrap::ErrorKind);
            Phonopy(::rsp2_phonopy_io::Error, ::rsp2_phonopy_io::ErrorKind);
            ExactLs(::rsp2_minimize::exact_ls::Error, ::rsp2_minimize::exact_ls::ErrorKind);
        }

        errors {
            /// Returned by the `from_existing()` methods of various Dir types.
            MissingFile(ty: &'static str, dir: PathBuf, filename: String) {
                description("Directory is missing a required file"),
                display("Directory '{}' is missing required file '{}' for '{}'",
                    dir.display(), &filename, ty),
            }
            NonPrimitiveStructure {
                description("attempted to compute symmetry of a supercell"),
                display("attempted to compute symmetry of a supercell"),
            }
            PhonopyFailed(status: ::std::process::ExitStatus) {
                description("phonopy exited unsuccessfully"),
                display("phonopy exited unsuccessfully ({})", status),
            }
        }
    }
    // fewer type annotations...
    pub fn ok<T>(x: T) -> Result<T> { Ok(x) }
    pub use ::std::result::Result as StdResult;
    pub use ::std::io::Result as IoResult;

    // so that CLI stubs don't need to import traits from error_chain
    // (why doesn't error_chain generate inherent method wrappers around this trait?)
    use error_chain::ChainedError;
    pub use error_chain::DisplayChain;
    impl Error {
        pub fn display_chain(&self) -> DisplayChain<Self>
        { ChainedError::display_chain(self) }

        pub fn is_missing_file(&self) -> bool
        { match *self {
            Error(ErrorKind::MissingFile(_, _, _), _) => true,
            _ => false,
        }}
    }
}


/// This module only exists to have its name appear in logs.
/// It marks a process's stdout.
mod stdout {
    pub fn log(s: &str)
    { info!("{}", s) }
}

/// This module only exists to have its name appear in logs.
/// It marks a process's stderr.
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
