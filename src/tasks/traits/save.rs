use ::traits::AsPath;
use ::FailResult;

use ::traits::IsNewtype;
use ::path_abs::{FileRead, FileWrite};

/// Uniform-ish "load a file" API for use by the highest level code (cmd).
/// Kinda technical debt now.
///
/// The idea here was to clean up ::cmd and phonopy::cmd by providing a
/// simple single-function interface for reading a value from a filepath
/// (i.e. the 90% use case for opening a file from the filesystem),
/// to hide away all of the (largely repetitive) details for composing all of the
/// orthogonal APIs for opening files, dealing with text vs binary, and
/// parsing/serialization.
///
/// I'm not sure whether it has been successful at this role. It has only made
/// `phonopy::cmd` a tad cleaner, and there's a decent number of things that simply
/// cannot use it, such as POSCAR files (because the values that are read/written
/// are asymmetric, and it's not nice to have to clone a structure into a wrapper
/// type just to be written).
///
/// ...as for now, it is what it is.
pub trait Load: Sized {
    fn load(path: impl AsPath) -> FailResult<Self>;
}

/// Load, but in the other direction.
pub trait Save {
    fn save(&self, path: impl AsPath) -> FailResult<()>;
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
    fn load(path: impl AsPath) -> FailResult<Json<T>>
    {Ok(::serde_json::from_reader(FileRead::read(path.as_path())?)?).map(Json)}
}

// Direct parsing of yaml must be done in extreme moderation due to compile times.
// impl<T> Load for Yaml<T> where T: for<'de> ::serde::Deserialize<'de> {
//     fn load<P: AsPath>(path: P) -> FailResult<Yaml<T>>
//     {Ok(::serde_yaml::from_reader(open(path.as_path())?)?).map(Yaml)}
// }

impl<T> Save for Json<T> where T: ::serde::Serialize {
    fn save(&self, path: impl AsPath) -> FailResult<()>
    {Ok(::serde_json::to_writer(FileWrite::create(path.as_path())?, &self.0)?)}
}

impl<T> Save for Yaml<T> where T: ::serde::Serialize {
    fn save(&self, path: impl AsPath) -> FailResult<()>
    {Ok(::serde_yaml::to_writer(FileWrite::create(path.as_path())?, &self.0)?)}
}
