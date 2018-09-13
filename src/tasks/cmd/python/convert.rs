/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

//! (spglib may be written in C, but I've had enough FFI.
//!  We're just going to call a python script.)

use ::FailResult;
use ::std::path::Path;
use ::traits::AsPath;

use super::{call_script_and_communicate, Script};

const PY_CALL_DYNMAT: Script = Script::Module("rsp2.internals.convert.dynmat");

#[derive(Serialize)]
struct Input<'a> {
    keep: bool,
    input: &'a Path,
    output: &'a Path,
}

#[allow(unused)]
pub enum Mode {
    Keep,
    Delete,
}

pub fn dynmat(
    input_path: impl AsPath,
    output_path: impl AsPath,
    mode: Mode,
) -> FailResult<()> {
    let input = Input {
        input: input_path.as_path(),
        output: output_path.as_path(),
        keep: match mode {
            Mode::Keep => true,
            Mode::Delete => false,
        },
    };
    call_script_and_communicate(PY_CALL_DYNMAT, &input)
}
