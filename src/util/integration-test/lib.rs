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

extern crate rsp2_fs_util as fsx;
extern crate failure;
extern crate path_abs;
#[cfg(feature = "test-diff")]
#[macro_use]
extern crate pretty_assertions;

use self::fsx::TempDir;
use self::path_abs::{PathDir, FileWrite, FileRead};
use self::failure::Error;

use ::std::fmt::Debug;
use ::std::fs::File;
use ::std::path::Path;
use ::std::ffi::{OsStr, OsString};
use ::std::process::Command;
pub type Result<T> = ::std::result::Result<T, Error>;

#[must_use]
pub struct CliTest {
    cmd: Vec<OsString>,
    expect_success: Option<bool>,
    checkers: Vec<DirChecker>,
}

impl Default for CliTest {
    fn default() -> Self {
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
        }
    }
}

pub type DirChecker = Box<Fn(&PathDir) -> Result<()>>;

impl CliTest {
    pub fn cargo_binary(name: impl AsRef<OsStr>) -> Self {
        let manifest_dir = PathDir::current_dir().unwrap();

        let test = CliTest {
            cmd: vec![],
            ..Self::default()
        };

        let test = test.arg("cargo").arg("run");
        #[cfg(not(debug_assertions))]
        let test = test.arg("--release");
        test.arg("--manifest-path")
            .arg(manifest_dir.join("Cargo.toml").as_path())
            .arg("--bin")
            .arg(name)
            .arg("--")
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
            check_against_with_diff(&actual, &expected, other.clone());
            Ok(())
        };
        self.check(checker)
    }

    pub fn run(self) -> Result<()> {
        let CliTest { cmd, checkers, expect_success } = self;

        let _tmp = TempDir::new("rsp2")?;
        let tmp = PathDir::new(_tmp.path())?;

        PathDir::current_dir()?.join("tests/resources").absolute()?.into_dir()?.symlink(tmp.join("resources"))?;

        let stdout_path = tmp.join("__captured_stdout");
        let stderr_path = tmp.join("__captured_stderr");
        let status = {
            let mut args = cmd;
            let bin = args.remove(0);

            Command::new(&bin)
                .args(&args)
                .current_dir(&tmp)
                // capture for the test harness
                .stdout({ let f: File = FileWrite::create(&stdout_path)?.into(); f})
                .stderr({ let f: File = FileWrite::create(&stderr_path)?.into(); f})
                .status()?
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

pub trait CheckFile: Sized + Debug + PartialEq + ::std::panic::RefUnwindSafe {
    // (Clone because checkers are Fn (can't dynamically call FnOnce) and the
    //  standard checker needs it)
    type OtherArgs: ::std::panic::UnwindSafe + Clone + 'static;

    fn read_file(path: &Path) -> Result<Self>;
    fn check_against(&self, expected: &Self, other_args: Self::OtherArgs);
}

#[cfg(feature = "test-diff")]
fn check_against_with_diff<T: CheckFile>(a: &T, b: &T, other: T::OtherArgs) {
    // Let check_against use things like `assert_close!` that might panic.
    let result = ::std::panic::catch_unwind(move || a.check_against(b, other));

    // If it did panic, throw that panic away and get a colorful character diff
    // on the Debug output from pretty_assertions.  This can help one get a quick
    // overview of how many decimal places are accurate throughout the entire file.
    if let Err(_) = result {
        let mention_save_tmp = match ::std::env::var("RSP2_SAVETEMP") {
            Err(::std::env::VarError::NotPresent) => {
                "  If the change looks reasonable, use RSP2_SAVETEMP=some-location \
                to recover the failed tempdir and copy the new file over into tests/resources."
            },
            _ => "",
        };
        assert_eq!(
            a, b,
            "Showing diff from pretty_assertion.{}", mention_save_tmp,
        );
        panic!("check_against failed but assert_eq succeeded?!");
    }
}

#[cfg(not(feature = "test-diff"))]
fn check_against_with_diff<T: CheckFile>(a: &T, b: &T, other: T::OtherArgs) {
    a.check_against(b, other)
}
