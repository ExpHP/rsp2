extern crate rsp2_kets;
extern crate rsp2_structure;
extern crate rsp2_array_types;

#[macro_use] extern crate failure;
#[macro_use] extern crate nom;
#[macro_use] extern crate serde_derive;
extern crate byte_tools;
extern crate serde_json;
extern crate serde_yaml;

pub use self::filetypes::{conf, Conf};
pub use self::filetypes::symmetry_yaml::{self, SymmetryYaml};
pub use self::filetypes::disp_yaml::{self, DispYaml};
pub use self::filetypes::force_sets::{self, ForceSets};

mod filetypes;
pub mod npy;

pub type FailResult<T> = Result<T, ::failure::Error>;
