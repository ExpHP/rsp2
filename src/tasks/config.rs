use ::rsp2_array_utils::vec_from_fn;
use ::rsp2_structure::Lattice;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Settings {
    pub supercell_relax: SupercellSpec,
    pub supercell_phonopy: SupercellSpec,
    pub symmetry_tolerance: f64, // 1e-5
    pub displacement_distance: f64, // 1e-3
    pub neg_frequency_threshold: f64, // 1e-3
    pub hack_scale: [f64; 3], // HACK
    pub layers: Option<u32>, // Number of layers, when known in advance
    pub cg: ::rsp2_minimize::acgsd::Settings,
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
                vec_from_fn(|k| {
                    (targets[k] / unit_lengths[k]).ceil().max(1.0) as u32
                })
            },
        }
    }
}
