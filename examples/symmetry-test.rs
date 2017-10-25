extern crate rsp2_tasks;
#[macro_use]
extern crate clap;
extern crate serde_yaml;

fn main() {

    let matches = clap_app!(myapp =>
        (version: "negative 0.00.3-734.bubbles")
        (author: "Michael T. Lamparski")
        (about: "blah")
        (@arg INPUT: +required "POSCAR")
    ).get_matches();
    let input = matches.value_of("INPUT").unwrap();

    let _ = ::rsp2_tasks::run_symmetry_test(input.as_ref());
}
