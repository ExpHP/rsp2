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
use crate::meta;
use crate::traits::{Save, Load, AsPath};
use crate::hlist_aliases::*;
use rsp2_structure_io::Poscar;
use rsp2_structure::Coords;
use crate::traits::save::Json;

use path_abs::PathDir;

const FNAME_STRUCTURE: &'static str = "POSCAR";
const FNAME_META: &'static str = "meta.json";
const FNAME_FRAC_BONDS: &'static str = "frac-bonds.json";

pub type Meta = HList5<
    meta::SiteElements,
    meta::SiteMasses,
    Option<meta::SiteLayers>,
    Option<meta::LayerScMatrices>,
    Option<meta::FracBonds>,
>;

/// "Filetype" for a structure that uses a directory.
///
/// Encodes as much information as possible, including things derived from config.
/// It is a suitable format for using the output of one run as input to another.
pub struct StoredStructure {
    pub title: String,
    pub coords: Coords,
    pub elements: meta::SiteElements,
    pub layers: Option<meta::SiteLayers>,
    pub masses: meta::SiteMasses,
    pub layer_sc_matrices: Option<meta::LayerScMatrices>,
    pub frac_bonds: Option<meta::FracBonds>,
}

impl StoredStructure {
    pub fn from_parts(title: impl Into<String>, coords: Coords, meta: Meta) -> Self {
        let hlist_pat![elements, masses, layers, layer_sc_matrices, frac_bonds] = meta;
        let title = title.into();
        StoredStructure {
            title, coords, elements, layers, masses, layer_sc_matrices, frac_bonds,
        }
    }
    pub fn meta(&self) -> Meta { hlist![
        self.elements.clone(), self.masses.clone(), self.layers.clone(),
        self.layer_sc_matrices.clone(), self.frac_bonds.clone(),
    ]}
    pub fn path_is_structure(path: impl AsPath) -> bool {
        path.join(FNAME_META).exists() && path.join(FNAME_STRUCTURE).exists()
    }
}

#[derive(Serialize, Deserialize)]
struct MetaJson {
    pub layers: Option<meta::SiteLayers>,
    pub masses: meta::SiteMasses,
    pub layer_sc_matrices: Option<meta::LayerScMatrices>,
}

impl Save for StoredStructure {
    fn save(&self, dir: impl AsPath) -> FailResult<()>
    {
        let dir = PathDir::create(dir.as_path())?; // (does not fail on existing directories)
        let StoredStructure {
            title, coords, elements, layers, masses, layer_sc_matrices, frac_bonds,
        } = self;

        Poscar { comment: title, coords, elements }.save(dir.join(FNAME_STRUCTURE))?;
        let layers = layers.clone();
        let masses = masses.clone();
        let layer_sc_matrices = layer_sc_matrices.clone();

        Json(MetaJson { layers, masses, layer_sc_matrices }).save(dir.join(FNAME_META))?;

        if let Some(frac_bonds) = frac_bonds {
            Json(frac_bonds).save(dir.join(FNAME_FRAC_BONDS))?;
        } else if dir.join(FNAME_FRAC_BONDS).exists() {
            let _ = std::fs::remove_file(dir.join(FNAME_FRAC_BONDS));
        }

        Ok(())
    }
}

impl Load for StoredStructure {
    fn load(dir: impl AsPath) -> FailResult<Self>
    {
        let dir = PathDir::new(dir.as_path())?;

        let Poscar { comment: title, coords, elements } = Load::load(dir.join(FNAME_STRUCTURE))?;
        let Json(meta) = Load::load(dir.join(FNAME_META))?;
        let MetaJson { layers, masses, layer_sc_matrices } = meta;
        let frac_bonds = if dir.join(FNAME_FRAC_BONDS).exists() {
            let Json(bonds) = Load::load(dir.join(FNAME_FRAC_BONDS))?;
            Some(bonds)
        } else { None };

        let elements = elements.into();
        Ok(StoredStructure { title, coords, masses, elements, layers, layer_sc_matrices, frac_bonds })
    }
}
