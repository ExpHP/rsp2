/* ********************************************************************** **
**  This file is part of rsp2.                                            **
**                                                                        **
**  rsp2 is free software: you can redistribute it and/or modify it under **
**  the terms of the GNU General Public License as published by the Free  **
**  Software Foundation, either version 3 of the License, or (at your     **
**  option) any later version.                                            **
**                                                                        **
**      http://www.gnu.org/licenses/                                      **
**                                                                        **
** Do note that, while the whole of rsp2 is licensed under the GPL, many  **
** parts of it are licensed under more permissive terms.                  **
** ********************************************************************** */

//! Implements the entry points for rsp2's binaries.

// HERE BE DRAGONS
//
// Basically, everything in this crate is purely ad-hoc.

#![deny(unused_must_use)]

extern crate rsp2_tasks_config;

extern crate rsp2_lammps_wrap;
extern crate rsp2_minimize;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_phonopy_io;
extern crate rsp2_array_utils;
extern crate rsp2_array_types;
extern crate rsp2_soa_ops;
extern crate rsp2_slice_math;
extern crate rsp2_fs_util;
extern crate rsp2_linalg;
#[macro_use] extern crate rsp2_newtype_indices;
#[macro_use] extern crate rsp2_util_macros;
#[macro_use] extern crate rsp2_clap;
#[macro_use] extern crate rsp2_assert_close;

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
extern crate serde_ignored;
extern crate serde_yaml;
extern crate env_logger;
extern crate num_traits;
#[macro_use] extern crate serde_json;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate failure;

extern crate lapacke;
extern crate lapack_src;

macro_rules! ichain {
    ($e:expr,) => { $e.into_iter() };
    ($e:expr, $($es:expr,)+)
    => { $e.into_iter().chain(ichain!($($es,)+)) };
}

// FIXME copied from failure 1.0 prerelease; remove once actually released
macro_rules! throw {
    ($e:expr) => {
        return Err(::std::convert::Into::into($e));
    }
}

#[macro_use]
mod util;
#[macro_use]
mod traits;
mod cmd;
mod phonopy;
mod math;
mod ui;
pub mod meta;
mod potential;
mod filetypes;

pub mod entry_points;

pub type FailResult<T> = Result<T, ::failure::Error>;
#[allow(bad_style)]
pub fn FailOk<T>(x: T) -> FailResult<T> { Ok(x) }
pub use ::std::io::Result as IoResult;

mod errors {
    use std::fmt;

    #[derive(Debug, Clone)]
    pub struct DisplayPathArcNice(pub ::path_abs::PathArc);
    impl fmt::Display for DisplayPathArcNice {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            use ::util::ext_traits::PathNiceExt;
            fmt::Display::fmt(&self.0.as_path().nice(), f)
        }
    }

}

/// This module only exists to have its name appear in logs.
/// It marks a child process's stdout.
mod stdout {
    use ::FailResult;
    use ::std::{process::ChildStdout, thread, io::{BufReader, BufRead}};
    use ::log::Level;

    const LEVEL: Level = Level::Info;

    pub fn log(s: &str)
    { log!(LEVEL, "{}", s) }

    #[allow(unused)]
    pub fn is_log_enabled() -> bool
    { log_enabled!(LEVEL) }

    pub fn spawn_log_worker(stdout: ChildStdout) -> thread::JoinHandle<FailResult<()>> {
        let f = BufReader::new(stdout);
        thread::spawn(move || -> ::FailResult<()> {Ok({
            for line in f.lines() {
                log(&(line?[..]));
            }
        })})
    }
}

/// This module only exists to have its name appear in logs.
/// It marks a child process's stderr.
mod stderr {
    use ::FailResult;
    use ::std::{process::ChildStderr, thread, io::{BufReader, BufRead}};
    use ::log::Level;

    const LEVEL: Level = Level::Warn;

    pub fn log(s: &str)
    { log!(LEVEL, "{}", s) }

    pub fn is_log_enabled() -> bool
    { log_enabled!(LEVEL) }

    pub fn spawn_log_worker(stderr: ChildStderr) -> thread::JoinHandle<FailResult<()>> {
        let f = BufReader::new(stderr);
        thread::spawn(move || -> ::FailResult<()> {Ok({
            for line in f.lines() {
                log(&(line?[..]));
            }
        })})
    }
}

mod env {
    use super::*;
    use ::std::env;
    use ::util::ext_traits::OptionResultExt;

    fn var(key: &str) -> FailResult<Option<String>>
    { match ::std::env::var(key) {
        Ok(s) => Ok(Some(s)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(s)) => bail!("env var not unicode: {}={:?}", key, s),
    }}

    fn nonempty_var(key: &str) -> FailResult<Option<String>>
    { match var(key) {
        Ok(Some(ref s)) if s == "" => Ok(None),
        r => r,
    }}

    pub fn rust_log() -> FailResult<String>
    {Ok({
        var("RUST_LOG")?.unwrap_or(String::new())
    })}

    /// Show module names in log output.
    pub fn log_mod() -> FailResult<bool>
    {Ok({
        nonempty_var("RSP2_LOG_MOD")?
            .map(|s| match &s[..] {
                "1" => Ok(true),
                "0" => Ok(false),
                _ => bail!("Invalid setting for RSP2_LOG_MOD: {:?}", s),
            }).fold_ok()?
            .unwrap_or(false)
    })}
}

mod common {
    use ::FailResult;
    use ::rsp2_structure::{Element, consts};

    pub fn default_element_mass(elem: Element) -> FailResult<f64>
    {Ok({
        match elem {
            consts::HYDROGEN => 1.00794,
            consts::CARBON => 12.0107,
            _ => bail!("No default mass for element {}.", elem.symbol()),
        }
    })}
}

// Although frunk has the variadic HList![] type macro, IntelliJ Rust can't handle it,
// with recent versions of the plugin painting large swathes of code in red syntax errors.
#[allow(unused)]
mod hlist_aliases {
    use ::frunk::{HNil, HCons};
    pub type HList0 = HNil;
    pub type HList1<A> = HCons<A, HList0<>>;
    pub type HList2<A, B> = HCons<A, HList1<B>>;
    pub type HList3<A, B, C> = HCons<A, HList2<B, C>>;
    pub type HList4<A, B, C, D> = HCons<A, HList3<B, C, D>>;
}

pub mod exposed_for_testing {
    pub use ::math::dynmat::ForceConstants;
    pub use ::meta;
}
