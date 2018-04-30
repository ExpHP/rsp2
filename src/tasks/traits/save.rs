use ::traits::AsPath;
use ::FailResult;

use ::traits::IsNewtype;
use ::path_abs::{FileRead, FileWrite};

pub trait Load: Sized {
    fn load<P>(path: P) -> FailResult<Self> where P: AsPath;
}

pub trait Save {
    fn save<P>(&self, path: P) -> FailResult<()> where P: AsPath;
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
    fn load<P: AsPath>(path: P) -> FailResult<Json<T>>
    {Ok(::serde_json::from_reader(FileRead::read(path.as_path())?)?).map(Json)}
}

// Direct parsing of yaml must be done in extreme moderation due to compile times.
// impl<T> Load for Yaml<T> where T: for<'de> ::serde::Deserialize<'de> {
//     fn load<P: AsPath>(path: P) -> FailResult<Yaml<T>>
//     {Ok(::serde_yaml::from_reader(open(path.as_path())?)?).map(Yaml)}
// }

impl<T> Save for Json<T> where T: ::serde::Serialize {
    fn save<P: AsPath>(&self, path: P) -> FailResult<()>
    {Ok(::serde_json::to_writer(FileWrite::create(path.as_path())?, &self.0)?)}
}

impl<T> Save for Yaml<T> where T: ::serde::Serialize {
    fn save<P: AsPath>(&self, path: P) -> FailResult<()>
    {Ok(::serde_yaml::to_writer(FileWrite::create(path.as_path())?, &self.0)?)}
}
