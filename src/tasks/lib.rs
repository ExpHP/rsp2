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
#![allow(non_snake_case)]
#![recursion_limit="128"]

#[macro_use] extern crate rsp2_newtype_indices;
#[macro_use] extern crate rsp2_util_macros;
#[macro_use] extern crate rsp2_clap;
#[macro_use] extern crate rsp2_assert_close;

#[macro_use] extern crate extension_trait;
#[macro_use] extern crate enum_map;
#[macro_use] extern crate frunk;
#[macro_use] extern crate indoc;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate failure;
extern crate lapacke;

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
pub mod math;
mod ui;
pub mod meta;
mod potential;
mod filetypes;
mod env;

pub mod entry_points;

pub type FailResult<T> = Result<T, failure::Error>;
#[allow(bad_style)]
pub fn FailOk<T>(x: T) -> FailResult<T> { Ok(x) }
pub use std::io::Result as IoResult;

mod errors {
    use std::fmt;

    #[derive(Debug, Clone)]
    pub struct DisplayPathNice(pub std::path::PathBuf);
    impl fmt::Display for DisplayPathNice {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            use crate::util::ext_traits::PathNiceExt;
            fmt::Display::fmt(&self.0.as_path().nice(), f)
        }
    }
}

/// This module only exists to have its name appear in logs.
/// It marks a child process's stdout.
mod stdout {
    use crate::FailResult;
    use std::{process::ChildStdout, thread, io::{BufReader, BufRead}};
    use log::Level;

    const LEVEL: Level = Level::Info;

    pub fn log(s: &str)
    { log!(LEVEL, "{}", s) }

    #[allow(unused)]
    pub fn is_log_enabled() -> bool
    { log_enabled!(LEVEL) }

    pub fn spawn_log_worker(stdout: ChildStdout) -> thread::JoinHandle<FailResult<()>> {
        let f = BufReader::new(stdout);
        thread::spawn(move || -> crate::FailResult<()> {Ok({
            for line in f.lines() {
                log(&(line?[..]));
            }
        })})
    }
}

/// This module only exists to have its name appear in logs.
/// It marks a child process's stderr.
mod stderr {
    use crate::FailResult;
    use std::{process::ChildStderr, thread, io::{BufReader, BufRead}};
    use log::Level;

    const LEVEL: Level = Level::Warn;

    pub fn log(s: &str)
    { log!(LEVEL, "{}", s) }

    pub fn is_log_enabled() -> bool
    { log_enabled!(LEVEL) }

    pub fn spawn_log_worker(stderr: ChildStderr) -> thread::JoinHandle<FailResult<()>> {
        let f = BufReader::new(stderr);
        thread::spawn(move || -> crate::FailResult<()> {Ok({
            for line in f.lines() {
                log(&(line?[..]));
            }
        })})
    }
}

mod common {
    use crate::FailResult;
    use crate::meta::{Element, Mass};
    use rsp2_structure::{consts};

    pub fn default_element_mass(elem: Element) -> FailResult<Mass>
    {Ok(Mass({
        match elem {
            consts::HYDROGEN => 1.00794,
            consts::CARBON => 12.0107,
            _ => bail!("No default mass for element {}.", elem.symbol()),
        }
    }))}
}

// Although frunk has the variadic HList![] type macro, IntelliJ Rust can't handle it,
// with recent versions of the plugin painting large swathes of code in red syntax errors.
#[allow(unused)]
mod hlist_aliases {
    use frunk::{HNil, HCons};
    pub type HList0 = HNil;
    pub type HList1<A> = HCons<A, HList0<>>;
    pub type HList2<A, B> = HCons<A, HList1<B>>;
    pub type HList3<A, B, C> = HCons<A, HList2<B, C>>;
    pub type HList4<A, B, C, D> = HCons<A, HList3<B, C, D>>;
    pub type HList5<A, B, C, D, E> = HCons<A, HList4<B, C, D, E>>;
    pub type HList6<A, B, C, D, E, F> = HCons<A, HList5<B, C, D, E, F>>;
    pub type HList7<A, B, C, D, E, F, G> = HCons<A, HList6<B, C, D, E, F, G>>;
    pub type HList8<A, B, C, D, E, F, G, H> = HCons<A, HList7<B, C, D, E, F, G, H>>;
    pub type HList9<A, B, C, D, E, F, G, H, I> = HCons<A, HList8<B, C, D, E, F, G, H, I>>;
}

pub mod exposed_for_testing {
    // FIXME move tests
    pub use rsp2_dynmat::ForceConstants;
    pub use crate::meta;
}

mod threading {
    /// Custom boolean type for enabling/disabling parallelism.
    ///
    /// Used in argument lists of internal APIs that want to use parallelism, to give threads
    /// a visible presence in high-level orchestration code. (You might see
    /// `(threading == cfg::Threading::Lammps).into()`, or even just a literal `Threading::Parallel`,
    /// which is at least more evocative than `true`)
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Threading { Serial, Parallel }

    impl From<bool> for Threading {
        fn from(b: bool) -> Threading
        { match b {
            true => Threading::Parallel,
            false => Threading::Serial,
        }}
    }

    impl Threading {
        /// Runs a closure in an environment where rayon has possibly been modified
        /// to run everything in serial.
        ///
        /// The name is `maybe_serial` as opposed to `maybe_parallel` because the code
        /// you write inside it should look like parallel code!
        pub fn maybe_serial<T, F>(self, f: F) -> T
        where
            T: Send,
            F: FnOnce() -> T + Send,
        { match self {
            Threading::Parallel => f(),
            Threading::Serial => {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap()
                    .install(f)
            },
        }}
    }
}

/// Version info, provided to rsp2-tasks by the entry points so that rsp2-tasks
/// itself does not need to be rebuilt.
///
/// (the function to generate it is in the toplevel crate `rsp2`)
#[derive(Debug, Copy, Clone)]
pub struct VersionInfo {
    pub short_sha: &'static str,
    pub commit_date: &'static str,
}
