use FailResult;
use meta::{Element, Layer, Mass};
use math::bands::ScMatrix;
use traits::{Save, Load, AsPath};

use rsp2_structure_io::Poscar;
use rsp2_structure::Coords;
use path_abs::PathDir;
use traits::save::Json;
use std::rc::Rc;

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
