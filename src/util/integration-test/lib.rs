
// NOTE: This draws heavily off of assert-cli (MIT 2.0/Apache)

extern crate rsp2_fs_util as fsx;
extern crate failure;
extern crate path_abs;

use self::fsx::TempDir;
use self::path_abs::{PathDir, FileWrite, FileRead};
use self::failure::Error;

use ::std::fs::File;
use ::std::ffi::{OsStr, OsString};
use ::std::process::Command;
pub type Result<T> = ::std::result::Result<T, Error>;

// thin layer that explodes on drop
// FIXME remove once we have must_use on functions
pub struct CliTest {
    inner: Option<CliTestInner>,
}

impl Drop for CliTest {
    fn drop(&mut self) {
        if !::std::thread::panicking() {
            panic!("CliTest was not run!")
        }
    }
}

pub struct CliTestInner {
    cmd: Vec<OsString>,
    expect_success: Option<bool>,
    checkers: Vec<DirChecker>,
}

impl Default for CliTest {
    fn default() -> Self {
        CliTest {
            inner: Some(CliTestInner {
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
            }),
        }
    }
}

pub type DirChecker = Box<Fn(&PathDir) -> Result<()>>;

impl CliTest {
    #[cfg_attr(feature = "nightly", must_use)]
    pub fn cargo_binary<S: AsRef<OsStr>>(name: S) -> Self {
        let manifest_dir = PathDir::current_dir().unwrap();

        let test = CliTest {
            inner: Some(CliTestInner {
                cmd: vec![],
                ..Self::default().into_inner()
            }),
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

    #[cfg_attr(feature = "nightly", must_use)]
    pub fn arg<S: AsRef<OsStr>>(mut self, arg: S) -> Self {
        self.inner.as_mut().unwrap().cmd.push(arg.as_ref().into());
        self
    }

    #[cfg_attr(feature = "nightly", must_use)]
    pub fn args<S: AsRef<OsStr>>(mut self, args: &[S]) -> Self {
        self.inner.as_mut().unwrap().cmd.extend(args.into_iter().map(OsString::from));
        self
    }

    #[cfg_attr(feature = "nightly", must_use)]
    pub fn check<F>(mut self, checker: F) -> Self
    where F: Fn(&PathDir) -> Result<()> + 'static,
    {
        self.inner.as_mut().unwrap().checkers.push(Box::new(checker));
        self
    }

    fn disarm(self) { ::std::mem::forget(self) }
    fn into_inner(mut self) -> CliTestInner {
        let inner = self.inner.take().unwrap();
        self.disarm();
        inner
    }

    #[cfg_attr(feature = "nightly", must_use)]
    pub fn run(self) -> Result<()> {
        let inner = self.into_inner();

        let _tmp = TempDir::new("rsp2")?;
        let tmp = PathDir::new(_tmp.path())?;

        PathDir::current_dir()?.join("tests/resources").absolute()?.into_dir()?.symlink(tmp.join("resources"))?;

        let stdout_path = tmp.join("__captured_stdout");
        let stderr_path = tmp.join("__captured_stderr");
        let status = {
            let mut args = inner.cmd;
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

        if let Some(success) = inner.expect_success {
            assert_eq!(success, status.success(), "{}", status);
        }

        for checker in inner.checkers {
            checker(&tmp)?;
        }
        Ok(())
    }
}

#[test]
#[should_panic(expected = "was not run")]
#[cfg(not(feature = "nightly"))]
fn cli_test_must_be_run() {
    CliTest::cargo_binary("lol");
}
