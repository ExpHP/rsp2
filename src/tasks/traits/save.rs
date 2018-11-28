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

use crate::FailResult;
use crate::traits::AsPath;
use crate::meta::Element;

use ::rsp2_structure::Coords;
use ::rsp2_structure_io::Poscar;
use ::path_abs::{FileRead, FileWrite};
use ::std::borrow::Borrow;
use ::std::io::BufReader;

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
/// cannot use it, where the input/output types are asymmetric.
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

/// Utility adapter for `Load`/`Save` that serializes as YAML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Yaml<T: ?Sized>(pub T);


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

impl<Comment, Coord, Elements> Save for Poscar<Comment, Coord, Elements>
where
    Comment: AsRef<str>,
    Coord: Borrow<Coords>,
    Elements: AsRef<[Element]>,
{
    fn save(&self, path: impl AsPath) -> FailResult<()>
    { Ok(self.to_writer(FileWrite::create(path.as_path())?)?) }
}

impl Load for Poscar {
    fn load(path: impl AsPath) -> FailResult<Poscar>
    {
        let file = BufReader::new(FileRead::read(path.as_path())?);
        Ok(Poscar::from_buf_reader(file)?)
    }
}
