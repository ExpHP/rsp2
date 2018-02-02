use ::Result;
use ::traits::{Save, Load, AsPath};
use ::rsp2_fs_util::{open, open_text, create};

use ::rsp2_phonopy_io::{symmetry_yaml, disp_yaml, conf};

mod cmd;
pub use self::cmd::*;

//--------------------------------------------------------

pub type SymmetryYaml = symmetry_yaml::SymmetryYaml;
impl Load for SymmetryYaml {
    fn load<P: AsPath>(path: P) -> Result<Self>
    { Ok(symmetry_yaml::read(open(path.as_path())?)?) }
}

//--------------------------------------------------------

pub type DispYaml = disp_yaml::DispYaml;
impl Load for DispYaml {
    fn load<P: AsPath>(path: P) -> Result<Self>
    { Ok(disp_yaml::read(open(path.as_path())?)?) }
}

//--------------------------------------------------------

// this is a type alias so we wrap it
#[derive(Debug, Clone, Default)]
pub struct Conf(pub ::rsp2_phonopy_io::Conf);
impl Load for Conf {
    fn load<P: AsPath>(path: P) -> Result<Self>
    { Ok(conf::read(open_text(path.as_path())?)?).map(Conf) }
}

impl Save for Conf {
    fn save<P: AsPath>(&self, path: P) -> Result<()>
    {Ok({ conf::write(create(path.as_path())?, &self.0)?; })}
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
    fn load<P: AsPath>(path: P) -> Result<Self>
    { Ok(::serde_json::from_reader(open(path.as_path())?)?) }
}

impl Save for Args {
    fn save<P: AsPath>(&self, path: P) -> Result<()>
    {Ok({ ::serde_json::to_writer(create(path.as_path())?, self)?; })}
}

//--------------------------------------------------------

/// Type representing Builder config that goes beyond simple adjustments to phonopy's input.
#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone)]
pub(crate) struct OtherSettings {
    use_sparse_sets: bool,
}

impl Load for OtherSettings {
    fn load<P: AsPath>(path: P) -> Result<Self>
    { Ok(::serde_json::from_reader(open(path.as_path())?)?) }
}

impl Save for OtherSettings {
    fn save<P: AsPath>(&self, path: P) -> Result<()>
    {Ok({ ::serde_json::to_writer(create(path.as_path())?, self)?; })}
}

impl OtherSettings {
    fn force_sets_filename(&self) -> &'static str
    { match self.use_sparse_sets {
        true => "SPARSE_SETS",
        false => "FORCE_SETS",
    }}
}

//--------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, Default)]
pub(crate) struct QPositions(Vec<[f64; 3]>);

impl Load for QPositions {
    fn load<P: AsPath>(path: P) -> Result<Self>
    { Ok(::serde_json::from_reader(open(path.as_path())?)?) }
}

impl Save for QPositions {
    fn save<P: AsPath>(&self, path: P) -> Result<()>
    {Ok({ ::serde_json::to_writer(create(path.as_path())?, self)?; })}
}
