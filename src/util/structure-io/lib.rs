extern crate rsp2_structure;
extern crate rsp2_array_types;

#[macro_use] extern crate itertools;
#[macro_use] extern crate error_chain;
extern crate vasp_poscar;

error_chain! {
    foreign_links {
        Io(::std::io::Error);
        ParseInt(::std::num::ParseIntError);
        Fail(::vasp_poscar::failure::Compat<::vasp_poscar::failure::Error>);
    }
}

pub mod poscar;
pub mod xyz;
