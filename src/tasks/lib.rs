//! Implements the entry points for rsp2's binaries.

// HERE BE DRAGONS
//
// Basically, everything in this crate is purely ad-hoc.

#![recursion_limit="256"] // for error chain...
#![deny(unused_must_use)]

extern crate rsp2_tasks_config;

extern crate rsp2_lammps_wrap;
extern crate rsp2_minimize;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_structure_gen;
extern crate rsp2_phonopy_io;
extern crate rsp2_array_utils;
extern crate rsp2_array_types;
extern crate rsp2_slice_math;
extern crate rsp2_fs_util;
#[macro_use] extern crate rsp2_util_macros;
#[macro_use] extern crate rsp2_clap;

#[macro_use] extern crate extension_trait;
#[macro_use] extern crate enum_map;
#[macro_use] extern crate frunk;
extern crate rayon;
extern crate rand;
extern crate slice_of_array;
extern crate serde;
#[macro_use] extern crate indoc;
extern crate ansi_term;
extern crate fern;
extern crate shlex;
#[macro_use] extern crate clap;
#[macro_use] extern crate lazy_static;
extern crate rsp2_kets;
extern crate path_abs;
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

macro_rules! _log_once_impl {
    ($mac:ident!($($arg:tt)*)) => {{
        use std::sync::{Once, ONCE_INIT};
        static ONCE: Once = ONCE_INIT;
        ONCE.call_once(|| {
            // Explicitly label one-time messages to discourage reasoning
            // along the lines of "well it didn't say anything THIS time"
            $mac!("{} (this message will not be shown again)", format_args!($($arg)*));
        });
    }};
}

macro_rules! warn_once { ($($arg:tt)*) => { _log_once_impl!{warn!($($arg)*)} }; }

#[macro_use]
mod traits;
mod util;
mod cmd;
mod phonopy;
mod math;
mod ui;

pub mod entry_points;

use errors::{Result, Error, ErrorKind, IoResult};
mod errors {
    use ::std::path::PathBuf;
    error_chain! {
        foreign_links {
            Io(::std::io::Error);
            Yaml(::serde_yaml::Error);
            Json(::serde_json::Error);
            SetLogger(::log::SetLoggerError);
            ParseInt(::std::num::ParseIntError);
            PathAbs(::path_abs::Error);
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

mod env {
    use super::*;
    use ::std::env;

    fn var(key: &str) -> Result<Option<String>>
    { match ::std::env::var(key) {
        Ok(s) => Ok(Some(s)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(s)) => bail!("env var not unicode: {}={:?}", key, s),
    }}

    /// Verbosity, as a signed integer env var.
    ///
    /// This is an env var for ease of implementation, so that the fern logger
    /// can be started eagerly rather than waiting until after we parse arguments.
    pub fn verbosity() -> Result<i32>
    {Ok({
        var("RSP2_VERBOSITY")?
            .unwrap_or_else(|| "0".into())
            .parse::<i32>()?
    })}
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

mod common {
    use ::rsp2_structure::{Element, consts};

    // FIXME: Should handle masses in a proper way (e.g. via metadata or otherwise) eventually...
    pub fn element_mass(elem: Element) -> f64
    {
        match elem {
            consts::HYDROGEN => 1.00794,
            consts::CARBON => 12.0107,
            _ => panic!("Missing mass for element {}.", elem.symbol()),
        }
    }
}
