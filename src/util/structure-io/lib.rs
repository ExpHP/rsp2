extern crate rsp2_structure;

extern crate itertools;
#[macro_use] extern crate error_chain;

error_chain! {
    foreign_links {
        Io(::std::io::Error);
    }
}

pub mod poscar;
pub mod xyz;
