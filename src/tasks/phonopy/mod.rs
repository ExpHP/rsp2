use ::FailResult;
use ::traits::{Save, Load, AsPath};
use ::traits::save::Json;
use ::errors::DisplayPathArc;

use ::rsp2_phonopy_io::{symmetry_yaml, disp_yaml, conf};
use ::std::io::BufReader;
use ::std::io::prelude::*;
use ::path_abs::{FileRead, FileWrite};
use ::rsp2_array_types::V3;
use ::failure::Backtrace;

mod cmd;
pub use self::cmd::*;

// Directory types in this module follow a pattern of having the datatype constructed
// after all files have been made; this is thrown when that is not upheld.
#[derive(Debug, Fail)]
#[fail(display = "Directory '{}' is missing required file '{}' for '{}'", dir, filename, ty)]
pub(crate) struct MissingFileError {
    backtrace: Backtrace,
    ty: &'static str,
    dir: DisplayPathArc,
    filename: String,
}

#[derive(Debug, Fail)]
#[fail(display = "phonopy failed with status {}", status)]
pub(crate) struct PhonopyFailed {
    backtrace: Backtrace,
    pub status: ::std::process::ExitStatus,
}

impl MissingFileError {
    fn new(ty: &'static str, dir: &AsPath, filename: String) -> Self {
        let backtrace = Backtrace::new();
        let dir = DisplayPathArc(dir.as_path().to_owned().into());
        MissingFileError { backtrace, ty, dir, filename }
    }
}

//--------------------------------------------------------

pub type SymmetryYaml = symmetry_yaml::SymmetryYaml;
impl Load for SymmetryYaml {
    fn load(path: impl AsPath) -> FailResult<Self>
    { Ok(symmetry_yaml::read(open(path.as_path())?)?) }
}

//--------------------------------------------------------

pub type DispYaml = disp_yaml::DispYaml;
impl Load for DispYaml {
    fn load(path: impl AsPath) -> FailResult<Self>
    { Ok(disp_yaml::read(open(path.as_path())?)?) }
}

//--------------------------------------------------------

// this is a type alias so we wrap it
#[derive(Debug, Clone, Default)]
pub struct Conf(pub ::rsp2_phonopy_io::Conf);
impl Load for Conf {
    fn load(path: impl AsPath) -> FailResult<Self>
    { Ok(conf::read(open_text(path.as_path())?).map(Conf)?) }
}

impl Save for Conf {
    fn save(&self, path: impl AsPath) -> FailResult<()>
    { Ok(conf::write(create(path.as_path())?, &self.0)?) }
}

//--------------------------------------------------------

/// Type representing extra CLI arguments.
///
/// Used internally to store things that must be preserved between
/// runs but cannot be set in conf files, like e.g. `--tolerance`
#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, Default)]
pub(crate) struct Args(Vec<String>);

impl<S, Ss> From<Ss> for Args
where
    S: AsRef<str>,
    Ss: IntoIterator<Item=S>,
{
    fn from(args: Ss) -> Self
    { Args(args.into_iter().map(|s| s.as_ref().to_owned()).collect()) }
}

impl Load for Args {
    fn load(path: impl AsPath) -> FailResult<Self>
    {
        use ::path_abs::FileRead;
        use ::util::ext_traits::PathNiceExt;
        let path = path.as_path();

        let text = FileRead::read(path)?.read_string()?;
        if let Some(args) = ::shlex::split(&text) {
            Ok(Args(args))
        } else {
            bail!("Bad args at {}", path.nice())
        }
    }
}

impl Save for Args {
    fn save(&self, path: impl AsPath) -> FailResult<()>
    {
        use ::path_abs::FileWrite;
        let mut file = FileWrite::create(path.as_path())?;
        for arg in &self.0 {
            writeln!(file, "{}", ::shlex::quote(arg))?;
        }
        Ok(())
    }
}

//--------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, Default)]
pub(crate) struct QPositions(Vec<V3>);

impl Load for QPositions {
    fn load(path: impl AsPath) -> FailResult<Self>
    { Load::load(path).map(|Json(x)| x) }
}

impl Save for QPositions {
    fn save(&self, path: impl AsPath) -> FailResult<()>
    { Json(self).save(path) }
}

//--------------------------------------------------------

fn open(path: impl AsPath) -> FailResult<FileRead>
{ Ok(FileRead::read(path.as_path())?) }

fn open_text(path: impl AsPath) -> FailResult<BufReader<FileRead>>
{ open(path).map(BufReader::new) }

fn create(path: impl AsPath) -> FailResult<FileWrite>
{ Ok(FileWrite::create(path.as_path())?) }
