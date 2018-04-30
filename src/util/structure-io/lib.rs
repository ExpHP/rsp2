extern crate rsp2_structure;
extern crate rsp2_array_types;

#[macro_use] extern crate itertools;
#[macro_use] extern crate failure;
extern crate vasp_poscar;

pub type FailResult<T> = Result<T, ::failure::Error>;

pub mod poscar;
pub mod xyz;
