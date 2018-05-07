#[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct RamanJson {
    pub frequency: Vec<f64>,
    pub average_3d: Vec<f64>,
    pub backscatter: Vec<f64>,
}
