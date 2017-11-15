extern crate rsp2_structure_gen;
extern crate rsp2_structure_io;
extern crate rsp2_structure;
#[macro_use]
extern crate clap;

use ::rsp2_structure::consts::CARBON;

fn main() {

    let matches = clap_app!(myapp =>
        (version: "negative 0.00.3-734.bubbles")
        (author: "Michael T. Lamparski")
        (about: "blah")
        (@arg LAYERS_YAML: +required "POSCAR")
    ).get_matches();
    let input = matches.value_of("LAYERS_YAML").unwrap();

    let mut a = ::rsp2_structure_gen::load_layers_yaml(::std::fs::File::open(input).unwrap()).unwrap();
    a.layer_seps()[0] = 20.0;
    a.vacuum_sep = 0.1;
    let s = a.assemble();
    ::rsp2_structure_io::poscar::dump(::std::io::stdout(), "", &s.map_metadata_into(|_| CARBON)).unwrap();
}
