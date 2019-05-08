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

// NOTE: This draws heavily off of assert-cli (MIT 2.0/Apache)

use crate::fsx::TempDir;
use path_abs::{PathDir, FileWrite, FileRead};
use failure::Error;

use std::fmt::Debug;
use std::fs::File;
use std::path::Path;
use std::ffi::{OsStr, OsString};
use std::process::Command;
pub type Result<T> = std::result::Result<T, Error>;

#[must_use]
pub struct CliTest {
    cmd: Vec<OsString>,
    expect_success: Option<bool>,
    checkers: Vec<DirChecker>,
    checker_index: usize, // where 'after_run's end and 'checkers' begin
}

pub type DirChecker = Box<dyn Fn(&PathDir) -> Result<()>>;
pub type AfterRun = Box<dyn Fn(&PathDir) -> Result<()>>;

impl CliTest {
    pub fn new(_env: &Environment) -> Self {
        CliTest {
            cmd: vec![
                "cargo",
                "run",
                #[cfg(not(debug_assertions))]
                "--release",
                "--quiet",
                "--",
            ].into_iter()
                .map(OsString::from)
                .collect(),
            expect_success: Some(true),
            checkers: vec![],
            checker_index: 0,
        }
    }

    pub fn cargo_binary(_env: &Environment, name: impl AsRef<OsStr>) -> Self {
        let manifest_dir = PathDir::current_dir().unwrap();

        let test = CliTest {
            cmd: vec![],
            ..Self::new(_env)
        };

        // Yep, we peek directly into `target`, because strangely enough this
        // is the proper thing to do.[citation needed]
        //
        // 'cargo run' might cause undesirable rebuilds of libraries/binaries that
        // were already built as dependencies of this integration test, due to ephemeral
        // changes in the environment inside cargo.
        //
        // Normally, running stuff directly from target/ would lead to linker issues
        // with missing dynamic library paths; but lucky for us, since cargo is running
        // this integration test, it has already modified `LD_LIBRARY_PATH` for us!
        let bin = {
            manifest_dir.join("target")
                .join(match cfg!(debug_assertions) {
                    true => "debug",
                    false => "release",
                })
                .join(name.as_ref())
        };
        let bin: &Path = bin.as_ref();
        test.arg(bin)
    }

    pub fn arg(mut self, arg: impl AsRef<OsStr>) -> Self {
        self.cmd.push(arg.as_ref().into());
        self
    }

    pub fn args<S: AsRef<OsStr>>(mut self, args: &[S]) -> Self {
        self.cmd.extend(args.into_iter().map(OsString::from));
        self
    }

    pub fn check<F>(mut self, checker: F) -> Self
    where F: Fn(&PathDir) -> Result<()> + 'static,
    {
        self.checkers.push(Box::new(checker));
        self
    }

    /// `check` with a standard trait-based implementation.
    pub fn check_file<T: CheckFile>(
        self,
        path_in_trial: &Path,
        expected_path: &Path,
        other: T::OtherArgs,
    ) -> Self {
        let path_in_trial = path_in_trial.to_owned();
        let expected_path = expected_path.to_owned();
        let checker = move |dir: &PathDir| {
            let actual = T::read_file(&dir.join(&path_in_trial))?;
            let expected = T::read_file(&expected_path)?;
            check_against_with_diff(&expected, &actual, other.clone());
            Ok(())
        };
        self.check(checker)
    }

    pub fn after_run<F>(mut self, callback: F) -> Self
    where F: Fn(&PathDir) -> Result<()> + 'static,
    {
        self.checkers.insert(self.checker_index, Box::new(callback));
        self.checker_index += 1;
        self
    }

    pub fn run(self) -> Result<()> {
        let CliTest { cmd, checkers, expect_success, checker_index: _ } = self;

        let _tmp = TempDir::new("rsp2")?;
        let tmp = PathDir::new(_tmp.path())?;

        let cwd = PathDir::current_dir()?;
        cwd.join("tests/resources").absolute()?.into_dir()?.symlink(tmp.join("resources"))?;

        // HACK to help tame useless rebuilds:
        //
        // .cargo/config is not strictly considered part of a cargo project;
        // It is merely the case that cargo checks all ancestors of the current directory.
        // This means changing directory into a tempdir can wreak havoc if you have a .cargo/config
        // that e.g. adds "-fopenmp".
        //
        // TODO: How does rsmpi manage to change linker arguments so easily?
        if let Ok(dot_cargo) = cwd.join(".cargo").absolute()?.into_dir() {
            dot_cargo.symlink(tmp.join(".cargo"))?;
        };

        let stdout_path = tmp.join("__captured_stdout");
        let stderr_path = tmp.join("__captured_stderr");
        let status = {
            let mut args = cmd;
            let bin = args.remove(0);

            let mut cmd = Command::new(&bin);
            cmd.args(&args);
            cmd.current_dir(&tmp);
            // capture for the test harness
            cmd.stdout({ let f: File = FileWrite::create(&stdout_path)?.into(); f});
            cmd.stderr({ let f: File = FileWrite::create(&stderr_path)?.into(); f});
            println!("Running {:?}", cmd);
            cmd.status()?
        };
        print!("{}", FileRead::read(stdout_path)?.read_string()?);
        eprint!("{}", FileRead::read(stderr_path)?.read_string()?);

        if let Some(success) = expect_success {
            assert_eq!(success, status.success(), "{}", status);
        }

        for checker in checkers {
            checker(&tmp)?;
        }
        Ok(())
    }
}

pub trait CheckFile: Sized + Debug + PartialEq + std::panic::RefUnwindSafe {
    // (Clone because checkers are Fn (can't dynamically call FnOnce) and the
    //  standard checker needs it)
    type OtherArgs: std::panic::UnwindSafe + Clone + 'static;

    fn read_file(path: &Path) -> Result<Self>;
    fn check_against(&self, expected: &Self, other_args: Self::OtherArgs);
}

#[cfg(feature = "test-diff")]
fn check_against_with_diff<T: CheckFile>(old: &T, new: &T, other: T::OtherArgs) {
    // Let check_against use things like `assert_close!` that might panic.
    let result = std::panic::catch_unwind(move || old.check_against(new, other));

    // If it did panic, throw that panic away and get a colorful character diff
    // on the Debug output from pretty_assertions.  This can help one get a quick
    // overview of how many decimal places are accurate throughout the entire file.
    if let Err(_) = result {
        let _guard = DoAfterPanic(|| {
            if std::env::var("RSP2_SAVETEMP") == Err(::std::env::VarError::NotPresent) {
                // Tell the user the next step towards blessing the new output.
                eprintln!();
                eprintln!("               The above panic should have printed a diff.");
                eprintln!("  If the change looks reasonable, use RSP2_SAVETEMP=some-location to recover");
                eprintln!("      the failed tempdir and copy the new file over into tests/resources.");
            }
        });
        // Use pretty_assertions
        assert_eq!(old, new);
    }
}

#[cfg(not(feature = "test-diff"))]
fn check_against_with_diff<T: CheckFile>(old: &T, new: &T, other: T::OtherArgs) {
    let _guard = DoAfterPanic(|| {
        // Tell the user the next step towards blessing the new output.
        eprintln!();
        eprintln!("  If the above panic occurred while comparing new output to old, ");
        eprintln!("    please try rerunning the test with '--features test-diff'.");
    });
    old.check_against(new, other)
}

// Call a function during unwind.
//
// It will be called after the panic handler (which is what usually prints the panic message and
// backtrace), making this a great way to ensure that a message appears at the very end of a
// test's captured STDERR.
struct DoAfterPanic<F: FnMut()>(F);

impl<F: FnMut()> Drop for DoAfterPanic<F> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            (self.0)()
        }
    }
}

/// Proof of the global environment for a test case having been set up.
pub struct Environment(());

static ENVIRONMENT_ONCE: std::sync::Once = std::sync::Once::new();

impl Environment {
    /// Set up the global environment for the test case.
    ///
    /// Most notably, this sets up a logger that prints to the captured stderr.
    ///
    /// Because this is used for test cases of CLI binaries, most of the log output
    /// you see in a test case is not actually from this (it is from the binary's
    /// stderr!).  However, this is needed to print the messages from `RSP2_SAVETEMP`
    /// that save the entire temp dir in which a test case was executed.
    pub fn init() -> Self {
        ENVIRONMENT_ONCE.call_once(|| {
            env_logger::Builder::new()
                .is_test(true)
                .filter_level(log::LevelFilter::Info)
                .init();
        });
        Environment(())
    }
}
