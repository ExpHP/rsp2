extern crate rsp2_structure;
extern crate rsp2_array_utils;
extern crate rsp2_array_types;
#[macro_use] extern crate rsp2_util_macros;

#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate failure;
#[macro_use] extern crate serde_derive;
extern crate serde;
extern crate serde_yaml;
extern crate vasp_poscar;

pub type FailResult<T> = Result<T, ::failure::Error>;
#[allow(bad_style)]
pub fn FailOk<T>(x: T) -> Result<T, ::failure::Error> { Ok(x) }

pub mod poscar;
pub mod xyz;
pub mod layers_yaml;
