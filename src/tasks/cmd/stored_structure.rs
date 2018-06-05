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

use ::FailResult;
use ::meta::{Element, Layer, Mass};
use ::math::bands::ScMatrix;
use ::traits::{Save, Load, AsPath};
use ::hlist_aliases::*;
use ::rsp2_structure_io::Poscar;
use ::rsp2_structure::Coords;
use ::traits::save::Json;

use ::path_abs::PathDir;
use ::std::rc::Rc;

/// "Filetype" for a structure that uses a directory.
///
/// Encodes as much information as possible, including things derived from config.
/// It is a suitable format for using the output of one run as input to another.
pub struct StoredStructure {
    pub title: String,
    pub coords: Coords,
    pub elements: Rc<[Element]>,
    pub layers: Option<Rc<[Layer]>>,
    pub masses: Rc<[Mass]>,
    pub layer_sc_matrices: Option<Vec<ScMatrix>>,
}

impl StoredStructure {
    pub fn meta(&self) -> HList3<Rc<[Element]>, Rc<[Mass]>, Option<Rc<[Layer]>>> {
        hlist![self.elements.clone(), self.masses.clone(), self.layers.clone()]
    }
}

#[derive(Serialize, Deserialize)]
struct StoredStructureMeta {
    pub layers: Option<Rc<[Layer]>>,
    pub masses: Rc<[Mass]>,
    pub layer_sc_matrices: Option<Vec<ScMatrix>>,
}

impl Save for StoredStructure {
    fn save(&self, dir: impl AsPath) -> FailResult<()>
    {
        let dir = PathDir::create(dir.as_path())?; // (does not fail on existing directories)
        let StoredStructure {
            title, coords, elements, layers, masses, layer_sc_matrices
        } = self;

        Poscar { comment: title, coords, elements }.save(dir.join("POSCAR"))?;
        let layers = layers.clone();
        let masses = masses.clone();
        let layer_sc_matrices = layer_sc_matrices.clone();

        Json(StoredStructureMeta { layers, masses, layer_sc_matrices })
            .save(dir.join("meta.json"))?;

        Ok(())
    }
}

impl Load for StoredStructure {
    fn load(dir: impl AsPath) -> FailResult<Self>
    {
        let dir = PathDir::new(dir.as_path())?;

        let Poscar { comment: title, coords, elements } = Load::load(dir.join("POSCAR"))?;
        let Json(meta) = Load::load(dir.join("meta.json"))?;
        let StoredStructureMeta { layers, masses, layer_sc_matrices } = meta;
        let elements = elements.into();
        Ok(StoredStructure { title, coords, masses, elements, layers, layer_sc_matrices })
    }
}
