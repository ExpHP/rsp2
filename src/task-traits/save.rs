use ::AsPath;
use ::Result;

use ::util::IsNewtype;
use ::rsp2_fs_util::{open, create};

pub trait Load: Sized {
    fn load<P>(path: P) -> Result<Self> where P: AsPath;
}

pub trait Save {
    fn save<P>(&self, path: P) -> Result<()> where P: AsPath;
}

/// Utility adapter for `Load`/`Save` that serializes as JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Json<T: ?Sized>(pub T);

unsafe impl<T: ?Sized> IsNewtype<T> for Json<T> { }

/// Utility adapter for `Load`/`Save` that serializes as YAML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Yaml<T: ?Sized>(pub T);

unsafe impl<T: ?Sized> IsNewtype<T> for Yaml<T> { }

impl<T> Load for Json<T> where T: for<'de> ::serde::Deserialize<'de> {
    fn load<P: AsPath>(path: P) -> Result<Json<T>>
    {Ok(::serde_json::from_reader(open(path.as_path())?)?).map(Json)}
}

impl<T> Load for Yaml<T> where T: for<'de> ::serde::Deserialize<'de> {
    fn load<P: AsPath>(path: P) -> Result<Yaml<T>>
    {Ok(::serde_yaml::from_reader(open(path.as_path())?)?).map(Yaml)}
}

impl<T> Save for Json<T> where T: ::serde::Serialize {
    fn save<P: AsPath>(&self, path: P) -> Result<()>
    {Ok(::serde_json::to_writer(create(path.as_path())?, &self.0)?)}
}

impl<T> Save for Yaml<T> where T: ::serde::Serialize {
    fn save<P: AsPath>(&self, path: P) -> Result<()>
    {Ok(::serde_yaml::to_writer(create(path.as_path())?, &self.0)?)}
}
