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

use rsp2_structure::{Coords, Element};
use rsp2_array_types::V3;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Hsd(String);

impl std::str::FromStr for Hsd {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // (we don't actually bother parsing HSD)
        Ok(Hsd(s.to_string()))
    }
}

#[derive(Debug)]
pub enum DftbPlus {}

#[derive(Debug, Clone)]
pub struct Builder {
    hsd: Hsd,
    elements: Option<Vec<Element>>,
    initial_coords: Option<Coords>,
    append_log: Option<PathBuf>,
}

impl Builder {
    pub fn from_hsd(hsd: &Hsd) -> Builder {
        Builder {
            hsd: hsd.clone(),
            initial_coords: None,
            elements: None,
            append_log: None,
        }
    }

    pub fn elements(&mut self, elements: &[Element]) -> &mut Self
    { self.elements = Some(elements.to_vec()); self }

    pub fn initial_coords(&mut self, coords: &Coords) -> &mut Self
    { self.initial_coords = Some(coords.clone()); self }

    pub fn append_log(&mut self, path: impl AsRef<Path>) -> &mut Self
    { self.append_log = Some(path.as_ref().to_owned()); self }

    pub fn build(&self) -> Result<DftbPlus, failure::Error> {
        unimplemented!()
    }
}

impl DftbPlus {
    pub fn elements(&self) -> &[Element] {
        match *self {}
    }

    pub fn set_coords(&mut self, _coords: Coords) -> Result<(), failure::Error> {
        match *self {}
    }

    pub fn compute_value(&mut self) -> Result<f64, failure::Error> {
        match *self {}
    }

    pub fn compute_grad(&mut self) -> Result<Vec<V3>, failure::Error> {
        match *self {}
    }
}
