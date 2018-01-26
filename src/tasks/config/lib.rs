// Crate where serde_yaml code is monomorphized, which is a huge compile time sink

// NOTE: Please make sure to use the YamlRead trait!
//       DO NOT USE serde_yaml::from_{reader,value,etc.} OUTSIDE THIS CRATE
//       or else you defeat the entire reason for its existence.

// (NOTE: I can't enforce this through the type system without completely destroying
//        the ergonomics of these types. Just Ctrl+Shift+F the workspace for "serde_yaml"
//        if compile times seem suspiciously off...)

#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;

extern crate serde;

extern crate rsp2_minimize;

use ::std::io::Read;
pub use ::rsp2_minimize::acgsd::Settings as Acgsd;

/// Provides an alternative to serde_yaml::from_reader where all of the
/// expensive codegen has already been performed in this crate.
pub trait YamlRead: for <'de> ::serde::Deserialize<'de> {
    fn from_reader<R: Read>(mut r: R) -> Result<Self, ::serde_yaml::Error>
    { YamlRead::from_dyn_reader(&mut r) }

    fn from_dyn_reader(r: &mut Read) -> Result<Self, ::serde_yaml::Error>;
}

macro_rules! derive_yaml_read {
    ($Type:ty) => {
        impl YamlRead for $Type {
            // NOTE: Moving this body into a default fn definition on the trait
            //       appears to make codegen lazy for some reason (compilation
            //       of this crate becomes suspiciously quick).
            //       Hence we generate these identical bodies in a macro.
            fn from_dyn_reader(r: &mut Read) -> Result<$Type, ::serde_yaml::Error>
            { ::serde_yaml::from_reader(r) }
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
    #[serde(default)]
    pub tweaks: Tweaks,
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
pub struct Tweaks {
    #[serde(default = "self::defaults::tweaks::sparse_sets")]
    pub sparse_sets: bool,
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

// --------------------------------------------------------

impl Default for Threading {
    fn default() -> Self { Threading::Lammps }
}

impl Default for EvLoop {
    fn default() -> Self { from_empty_mapping().unwrap() }
}
#[test] fn test_ev_loop_default() { let _ = EvLoop::default(); }

impl Default for Tweaks {
    fn default() -> Self { from_empty_mapping().unwrap() }
}
#[test] fn test_tweaks_default() { let _ = Tweaks::default(); }

fn from_empty_mapping<T: for<'de> ::serde::Deserialize<'de>>() -> ::serde_yaml::Result<T> {
    use ::serde_yaml::{from_value, Value, Mapping};
    from_value(Value::Mapping(Mapping::new()))
}

mod defaults {
    pub(crate) mod settings {
        pub(crate) fn min_positive_iters() -> u32 { 3 }
    }

    pub(crate) mod tweaks {
        pub(crate) fn sparse_sets() -> bool { false }
    }

    pub(crate) mod scale_ranges {
        pub(crate) fn warn() -> Option<f64> { Some(0.01) }
    }

    pub(crate) mod ev_loop {
        pub(crate) fn max_iter() -> u32 { 15 }
        pub(crate) fn fail() -> bool { true }
    }
}
