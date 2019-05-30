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
pub mod convert;

//---------------------------------------------------------
use crate::{FailResult};
use std::process;
use std::path::{Path, PathBuf};

use rsp2_fs_util as fsx;

const PY_NOOP: Script = Script::String(indoc!(r#"
    #!/usr/bin/env python3
"#));

// special log target for recording data written to stdout that was intended to be deserialized
mod raw_stdout_log {
    pub const TARGET: &str = "rsp2_tasks::special::py_stdout";
    pub fn enabled() -> bool { log_enabled!(target: TARGET, log::Level::Trace) }
    pub fn suggestion() -> String { format!("RUST_LOG={}=trace", TARGET) }
}

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

fn call_script_and_check_success<E: failure::Fail>(
    script: Script,
    error: E,
) -> FailResult<()>
{Ok({
    use std::process::Stdio;

    let tmp = fsx::TempDir::new("rsp2")?;
    let script = ReifiedScript::new(script, tmp.path().join("script.py"))?;

    let mut cmd = process::Command::new("python3");
    script.add_args(&mut cmd);

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let child_stdout = child.stdout.take().unwrap();
    let child_stderr = child.stderr.take().unwrap();

    let stdout_worker = crate::stdout::spawn_log_worker(child_stdout);
    let stderr_worker = crate::stderr::spawn_log_worker(child_stderr);

    if !child.wait()?.success() {
        throw!(error);
    }

    let _ = stdout_worker.join();
    let _ = stderr_worker.join();
})}

enum Script {
    /// Run a script saved as a python module, using `python -m fully.qualified.name`.
    ///
    /// This module is resolved through all of the standard means (installed packages, `PYTHONPATH`,
    /// etc...)
    Module(&'static str),
    /// Run a standalone script whose text is embedded in the binary, using `python tempfile.py`
    String(&'static str),
}

fn call_script_and_communicate<In, Out>(
    script: Script,
    stdin_data: In,
) -> FailResult<Out>
where
    In: serde::Serialize,
    Out: for<'de> serde::Deserialize<'de>,
{Ok({
    use std::process::Stdio;

    let tmp = fsx::TempDir::new("rsp2")?;
//    tmp.try_with_recovery(|tmp| FailOk({
        let script = ReifiedScript::new(script, tmp.path().join("script.py"))?;

        let mut cmd = process::Command::new("python3");
        script.add_args(&mut cmd);

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // FIXME HACK:
        // It'd be a PITA to propagate down threading configuration, but for now, I know
        // that rsp2 never runs more than one python script at at time.
        cmd.env("OMP_NUM_THREADS", crate::env::max_omp_num_threads()?.to_string());

        let mut child = cmd.spawn()?;
        let child_stdin = child.stdin.take().unwrap();
        let child_stderr = child.stderr.take().unwrap();
        let child_stdout = child.stdout.take().unwrap();

        let stderr_worker = crate::stderr::spawn_log_worker(child_stderr);

        serde_json::to_writer(child_stdin, &stdin_data)?;

        // for debugging deserialization errors
        let child_stdout = {
            let log_file = match raw_stdout_log::enabled() {
                true => fsx::create(tmp.path().join("_py_stdout"))?,
                false => fsx::create("/dev/null")?,
            };
            tee::TeeReader::new(child_stdout, log_file)
        };

        // Deserialization must be done before checking the exit code...
        let de_result = serde_json::from_reader(child_stdout);

        // ...but prioritize a nonzero exit status over an error in deserialization.
        if !child.wait()?.success() {
            let extra = match crate::stderr::is_log_enabled() {
                true => "check the log for a python backtrace",
                false => "that's all we know",
            };
            bail!("an error occurred in a python script; {}", extra);
        };
        let _ = stderr_worker.join();

        // Only if the script reported success do we care about deserialization errors.
        de_result.unwrap_or_else(move |de_error| {
            let extra = match raw_stdout_log::enabled() {
                false => format!("To record the ill-formed output, try setting {}", raw_stdout_log::suggestion()),
                true => format!("The ill-formed output was logged to a temp dir."),
            };
            info!("{}", tmp.path().display());
            let _ = crate::util::recover_temp_dir_if_non_empty(tmp);
            panic!("(BUG!) Error during deserialization: {}\n{}", de_error, extra);
        })
//    }))?.1 // tmp.try_with_recovery(...)
})}

/// The runtime component of Script.  Constructing it may produce a file on the filesystem,
/// and it knows what arguments must be handed to the python interpreter to invoke the script.
struct ReifiedScript {
    script: Script,
    file_path: Option<PathBuf>,
}

impl ReifiedScript {
    fn new(script: Script, suggested_path: impl AsRef<Path>) -> FailResult<Self> {
        let suggested_path = suggested_path.as_ref();

        let file_path = match script {
            Script::Module(_) => None,
            Script::String(s) => {
                fsx::write(suggested_path, s)?;
                Some(suggested_path.to_owned())
            },
        };
        Ok(ReifiedScript { script, file_path })
    }

    // Add the argument(s) to a `python` command specifying the script to run.
    //
    // Any arguments after these will be forwarded directly to the script.
    fn add_args(&self, cmd: &mut process::Command) {
        match self.script {
            Script::String(_) => cmd.arg(self.file_path.as_ref().unwrap()),
            Script::Module(s) => cmd.arg("-m").arg(s),
        };
    }
}
