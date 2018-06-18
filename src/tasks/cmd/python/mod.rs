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

pub use self::spglib::SpgDataset;
mod scipy_eigsh;
mod spglib;

//---------------------------------------------------------
use ::{FailResult, FailOk};
use ::std::process;
use ::std::io::prelude::*;

const PY_NOOP: &'static str = indoc!(r#"
    #!/usr/bin/env python3
"#);

#[derive(Debug, Fail)]
#[fail(display = "an error occurred running the most trivial python script")]
pub struct PythonExecutionError;

pub fn check_availability() -> FailResult<()> {
    use self::scipy_eigsh::PY_CHECK_SCIPY_AVAILABILITY;
    use self::scipy_eigsh::ScipyAvailabilityError;

    use self::spglib::PY_CHECK_SPGLIB_AVAILABILITY;
    use self::spglib::SpglibAvailabilityError;

    call_script_and_check_success(PY_NOOP, PythonExecutionError)?;
    call_script_and_check_success(PY_CHECK_SCIPY_AVAILABILITY, ScipyAvailabilityError)?;
    call_script_and_check_success(PY_CHECK_SPGLIB_AVAILABILITY, SpglibAvailabilityError)?;

    Ok(())
}

fn call_script_and_check_success<E: ::failure::Fail>(
    script: &'static str,
    error: E,
) -> FailResult<()>
{Ok({
    use ::std::process::Stdio;

    let tmp = ::rsp2_fs_util::TempDir::new("rsp2")?;
    let path = tmp.path().join("script.py");

    ::std::fs::write(&path, script)?;

    let mut cmd = process::Command::new("python3");
    cmd.arg(path);

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let child_stdout = child.stdout.take().unwrap();
    let child_stderr = child.stderr.take().unwrap();

    let stdout_worker = ::stdout::spawn_log_worker(child_stdout);
    let stderr_worker = ::stderr::spawn_log_worker(child_stderr);

    if !child.wait()?.success() {
        throw!(error);
    }

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

fn call_script_and_communicate<In, Out>(
    script: &'static str,
    other_modules: &[(&'static str, &'static str)],
    stdin_data: In,
) -> FailResult<Out>
where
    In: ::serde::Serialize,
    Out: for<'de> ::serde::Deserialize<'de>,
{Ok({
    use ::std::process::Stdio;

    let tmp = ::rsp2_fs_util::TempDir::new("rsp2")?;
    tmp.try_with_recovery(|tmp| FailOk({
        let main_path = tmp.path().join("main-script.py");
        ::std::fs::write(&main_path, script)?;

        for &(name, content) in other_modules {
            let path = tmp.path().join(format!("{}.py", name));
            ::std::fs::write(&path, content)?;
        }

        let mut cmd = process::Command::new("python3");
        cmd.arg(main_path);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;
        let child_stdin = child.stdin.take().unwrap();
        let child_stderr = child.stderr.take().unwrap();
        let mut child_stdout = child.stdout.take().unwrap();

        let stderr_worker = ::stderr::spawn_log_worker(child_stderr);

        ::serde_json::to_writer(child_stdin, &stdin_data)?;

        let stdout = {
            let mut buf = String::new();
            child_stdout.read_to_string(&mut buf)?;
            buf
        };
        // for debugging
        ::std::fs::write(tmp.path().join("_py_stdout"), &stdout)?;
        let value = ::serde_json::from_str(&stdout[..])?;

        if !child.wait()?.success() {
            let extra = match ::stderr::is_log_enabled() {
                true => "check the log for a python backtrace",
                false => "that's all we now",
            };
            bail!("an error occurred in a python script; {}", extra);
        }

        let _ = stderr_worker.join();

        value
    }))?.1 // tmp.try_with_recovery(...)
})}
