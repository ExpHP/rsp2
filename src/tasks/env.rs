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

use crate::FailResult;
use std::env;
use crate::util::ext_traits::OptionResultExt;

fn var(key: &str) -> FailResult<Option<String>>
{ match std::env::var(key) {
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

// (not necessarily an integer but may be a comma-separated list.
//  I'm sticking to a String as we only use it for display)
pub const OMP_NUM_THREADS: &'static str = "OMP_NUM_THREADS";
pub fn omp_num_threads() -> FailResult<String>
{
    nonempty_var(OMP_NUM_THREADS)
        .map(|s| s.unwrap_or_else(|| "1".into()))
}

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

pub const MAX_OMP_NUM_THREADS: &'static str = "RSP2_MAX_THREADS";
/// Maximum possible setting for OMP_NUM_THREADS for a single process (to use the full
/// computing power of its node).
///
/// Using MPI seems to somehow fool the `num_cpus` crate into seeing fewer cores.
/// This should be manually set to the number of cores on a node.
pub fn max_omp_num_threads() -> FailResult<u32>
{Ok({
    nonempty_var(MAX_OMP_NUM_THREADS)?
        .map(|s| s.parse())
        .fold_ok()?
        .unwrap_or(::num_cpus::get() as u32)
})}

#[cfg(feature = "mpi")]
pub fn num_mpi_processes() -> u32 {
    use mpi::traits::Communicator;

    let world = mpi::topology::SystemCommunicator::world();
    world.size() as _
}
