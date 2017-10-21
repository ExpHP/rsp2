extern crate rsp2_structure;

#[macro_use] extern crate itertools;
#[macro_use] extern crate error_chain;

error_chain! {
    foreign_links {
        Io(::std::io::Error);
        ParseInt(::std::num::ParseIntError);
    }
}

pub mod poscar;
pub mod xyz;
