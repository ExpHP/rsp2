extern crate rsp2_structure;
extern crate rsp2_array_utils;
extern crate rsp2_array_types;

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
// fewer type annotations
fn ok<T>(x: T) -> Result<T> { Ok(x) }

mod assemble;
pub use assemble::load_layers_yaml;
pub use assemble::Assemble;
// FIXME this really doesn't belong here, but it's the easiest reuse of code
pub use assemble::layer_sc_info_from_layers_yaml;
