// Crate where serde_yaml code is monomorphized, which is a huge compile time sink

#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;
#[macro_use]
extern crate serde_json; // FIXME only used to assist default impls
extern crate serde;

#[macro_use]
extern crate rsp2_util_macros; // FIXME only used to assist default impls
extern crate rsp2_array_utils;
extern crate rsp2_structure;
extern crate rsp2_minimize;

use ::rsp2_array_utils::arr_from_fn;
use ::rsp2_structure::Lattice;

pub use ::rsp2_minimize::acgsd::Settings as Acgsd;
pub use ::serde_yaml::Error as Error;
pub type Result<T> = ::std::result::Result<T, Error>;

pub trait YamlRead: for <'de> ::serde::Deserialize<'de> {
    fn read_from_bytes(bytes: &[u8]) -> Result<Self>;
}

macro_rules! derive_yaml_read {
    ($Type:ty) => {
        impl YamlRead for $Type {
            fn read_from_bytes(bytes: &[u8]) -> Result<Self>
            { ::serde_yaml::from_reader(bytes) }
        }
    };
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Settings {
    #[serde(default)]
    pub threading: Threading,
    pub potential: Potential,
    pub scale_ranges: ScaleRanges,
    // Number of layers, when known in advance
    pub layers: Option<u32>,
    pub cg: Acgsd,
    pub phonons: Phonons,
    pub ev_chase: EigenvectorChase,
    // Relaxation stops after all EVs are positive this many times
    #[serde(default = "self::defaults::settings::min_positive_iters")]
    pub min_positive_iters: u32,
    pub layer_gamma_threshold: f64,
    #[serde(default)]
    pub ev_loop: EvLoop,
}
derive_yaml_read!{Settings}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct ScaleRanges {
    pub parameter: ScaleRange,
    pub layer_sep: ScaleRange,
    #[serde(default="self::defaults::scale_ranges::warn")]
    pub warn: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ScaleRange {
    #[serde(rename_all = "kebab-case")]
    Range {
        range: (f64, f64),
        /// A "reasonable value" that might be used while another
        ///  parameter is optimized.
        guess: Option<f64>,
    },
    Exact(f64),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct EnergyPlotSettings {
    #[serde(default)]
    pub threading: Threading,
    pub xlim: [f64; 2],
    pub ylim: [f64; 2],
    pub dim: [usize; 2],
    pub ev_indices: EnergyPlotEvIndices,
    /// Defines scale of xlim/ylim.
    pub normalization: NormalizationMode,
    //pub phonons: Phonons,
    pub lj: LennardJones,
}
derive_yaml_read!{EnergyPlotSettings}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum EnergyPlotEvIndices {
    Shear,
    These(usize, usize),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Potential {
    // supercell for lammps.
    // Purpose is to help eliminate boundary effects or something?
    // I forget.  Might not be necessary
    pub supercell: SupercellSpec,
    pub lj: LennardJones,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct LennardJones {
    /// Cutoff radius (x3.4A)
    pub sigma: Option<f64>,
    /// Scale hack
    pub strength: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum EigenvectorChase {
    OneByOne,
    #[serde(rename = "cg")]
    Acgsd(Acgsd),
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Phonons {
    pub symmetry_tolerance: f64,
    pub displacement_distance: f64,
    pub supercell: SupercellSpec,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum SupercellSpec {
    Target([f64; 3]),
    Dim([u32; 3]),
}

impl SupercellSpec {
    pub fn dim_for_unitcell(&self, prim: &Lattice) -> [u32; 3] {
        match *self {
            SupercellSpec::Dim(d) => d,
            SupercellSpec::Target(targets) => {
                let unit_lengths = prim.lengths();
                arr_from_fn(|k| {
                    (targets[k] / unit_lengths[k]).ceil().max(1.0) as u32
                })
            },
        }
    }
}

// FIXME delete
#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all="kebab-case")]
pub enum Threading {
    Lammps,
    Rayon,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum NormalizationMode {
    /// Normalize the 2-norm of the 3N-component vector.
    CoordNorm,

    // These are anticipated but YAGNI for now.
    //    /// Normalize rms of the 3N-component vector to 1.
    //    CoordRms,
    //    /// Normalize mean of the 3N-component vector to 1.
    //    CoordMean,
    //    /// Normalize max value of the 3N-component vector to 1.
    //    CoordMax,

    /// Normalize rms atomic displacement distance to 1.
    AtomRms,
    /// Normalize mean atomic displacement distance to 1.
    AtomMean,
    /// Normalize max atomic displacement distance to 1.
    AtomMax,
}

#[derive(Serialize, Deserialize)]
#[derive(Debug, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct EvLoop {
    #[serde(default = "self::defaults::ev_loop::max_iter")]
    pub max_iter: u32,
    #[serde(default = "self::defaults::ev_loop::fail")]
    pub fail: bool,
}

impl Default for Threading {
    fn default() -> Self { Threading::Lammps }
}

impl Default for EvLoop {
    fn default() -> Self { from_json!({}) }
}

mod defaults {
    pub(crate) mod settings {
        pub(crate) fn min_positive_iters() -> u32 { 3 }
    }

    pub(crate) mod scale_ranges {
        pub(crate) fn warn() -> Option<f64> { Some(0.01) }
    }

    pub(crate) mod ev_loop {
        pub(crate) fn max_iter() -> u32 { 15 }
        pub(crate) fn fail() -> bool { true }
    }
}
