use rsp2_structure::bonds;
use rsp2_structure_io::layers_yaml;
use std::fs::File;

#[test]
fn giant_bonds() {
    let mut layers = layers_yaml::load(File::open("tests/19153-113-19154-1-1-1-layers.yaml").unwrap()).unwrap();
    layers.vacuum_sep = 20.0;
    layers.scale = [2.46, 2.46, 1.0];
    layers.layer_seps()[0] = 3.38;
    let coords = layers.assemble();

    for _ in 0..10 {
        let bonds = bonds::FracBonds::compute(&coords, 1.8).unwrap();
        let graph = bonds.to_periodic_graph();
        println!("{}", graph.edge_count());
    }
}
