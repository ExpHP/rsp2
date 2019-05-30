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

use crate::FailResult;
use crate::traits::AsPath;
use crate::math::dynmat::DynamicalMatrix;

use super::{call_script_and_communicate_with_args, Script};

const PY_CALL_READ_DYNMAT: Script = Script::Module("rsp2.internals.convert.read_dynmat");
const PY_CALL_WRITE_DYNMAT: Script = Script::Module("rsp2.internals.convert.write_dynmat");

pub fn read_dynmat(
    input_path: impl AsPath,
) -> FailResult<DynamicalMatrix> {
    let cereal = call_script_and_communicate_with_args(
        PY_CALL_READ_DYNMAT,
        &(),
        |cmd| { cmd.arg(input_path.as_path()); },
    )?;

    DynamicalMatrix::from_cereal(cereal)
}

pub fn write_dynmat(
    output_path: impl AsPath,
    dynmat: &DynamicalMatrix,
) -> FailResult<()> {
    call_script_and_communicate_with_args(
        PY_CALL_WRITE_DYNMAT,
        &dynmat.cereal(),
        |cmd| { cmd.arg(output_path.as_path()); },
    )
}
