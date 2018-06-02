//! (spglib may be written in C, but I've had enough FFI.
//!  We're just going to call a python script.)

use ::FailResult;
use ::meta::Element;
use ::meta::prelude::*;
use ::hlist_aliases::*;
use ::rsp2_array_types::{V3, M33};
use ::rsp2_structure::{Coords};

use ::std::rc::Rc;

use super::{call_script_and_communicate};

pub(super) const PY_CHECK_SPGLIB_AVAILABILITY: &'static str = indoc!(r#"
    #!/usr/bin/env python3
    import spglib
    spglib.get_symmetry_dataset
"#);

const PY_CALL_SPGLIB: &'static str = include_str!("call-spglib.py");

#[derive(Serialize)]
struct Input {
    coords: Coords,
    types: Vec<u32>,
    symprec: f64,
}

//-------------------------------------------------------------------------------
// calling scripts

#[derive(Debug, Fail)]
#[fail(display = "an error occurred importing the spglib python module")]
pub struct SpglibAvailabilityError;

//-------------------------------------------------------------------------------

impl SpgDataset {
    pub fn compute(coords: &Coords, types: &[u32], symprec: f64) -> FailResult<Self> {
        let mut coords = coords.clone();
        coords.ensure_fracs();
        let types = types.to_vec();

        let input = Input { coords, types, symprec };
        call_script_and_communicate(PY_CALL_SPGLIB, &input)
    }
}

#[derive(Deserialize)]
#[derive(Debug, Clone)]
pub struct SpgDataset {
    #[serde(rename = "number")]
    pub space_group_number: u32,
    #[serde(rename = "international")]
    pub international_symbol: String,

    pub pointgroup: String,

    pub hall_number: u32,
    #[serde(rename = "hall")]
    pub hall_symbol: String,

    // *shrug*
    pub choice: String,

    // it returns this as floats, though I'm not sure if it
    // should ever be non-integral...
    pub transformation_matrix: M33,

    /// To the best of my knowledge:
    ///
    /// * The rotations and translations are expressed in fractional units of the **input cell.**
    ///   They can be used as is on the input fractional coords without any preprocessing.
    ///   (I think)
    /// * If the input structure was a supercell, then **pure translations will be included.**
    /// * (aside: according to the spglib documentation, if the chosen supercell shape breaks some
    ///    symmetries, those symmetries will be omitted even though they would be valid physically;
    ///    This is by design)
    /// * Rotation is in a **coordinate-major layout** (as opposed to vector-major).
    ///   Alternatively, one might say that, if each inner list were regarded as a row vector,
    ///   you would have the matrix that operates on column vectors. (and vice versa)
    pub rotations: Vec<M33<i32>>,
    /// These are the translations that accompany each rotation.
    pub translations: Vec<V3>,

    /// A standard origin for rotation, in fractional coords.
    ///
    /// This can be used to recover an integer description of the translations.
    /// Basically, for a primitive cell, if the spacegroup operators are performed around
    /// this point rather than the origin, you will obtain operators where the translation
    /// coordinates are all multiples of `1/12`.  (I *think.*)
    ///
    /// This is true for any cell that is primitive (i.e. of minimal volume).
    ///
    /// (to clarify, you must construct the operation that translates by `-origin_shift`,
    ///  performs `rotation`, performs `translation`, and finally translate by `+origin_shift`.
    ///  The resulting operators are not symmetries of the input coords as written (you would
    ///  have to translate them), but they will share the group structure, and have translations
    ///  that follow the `1/12`th rule if the cell was primitive)
    pub origin_shift: V3,

    /// Wyckoff position of each atom.
    pub wyckoffs: Vec<String>,
    pub equivalent_atoms: Vec<usize>,
    pub mapping_to_primitive: Vec<usize>,

    // I think these describe BPOSCAR?
    pub std_lattice: M33,
    pub std_types: Vec<u32>,
    pub std_positions: Vec<V3>,
    pub std_mapping_to_primitive: Vec<usize>,
}