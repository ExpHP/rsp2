extern crate rsp2_structure;

extern crate serde;
extern crate serde_yaml;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate error_chain;

error_chain! {
    foreign_links {
        Io(::std::io::Error);
        Yaml(::serde_yaml::Error);
    }
}

mod assemble;
pub use assemble::load_layers_yaml;
