extern crate rsp2_tasks;
#[macro_use]
extern crate rsp2_clap;
use ::rsp2_clap::clap;

fn main() {
    let matches = ::clap::App::new("thing")
        .version("negative 0.00.3-734.bubbles")
        .author("Michael T. Lamparski")
        .about("blah")
        .args(&[
            arg!(*config [-c][--config]=CONFIG "settings yaml"),
            arg!( dir=DIR "Output dir from relaxation"),
        ])
        .get_matches();
    let dir = matches.value_of("dir").expect("bug! dir");
    let config = matches.value_of("config").expect("bug! config");
    let settings = ::rsp2_tasks::config::read_yaml(::std::fs::File::open(config).unwrap()).unwrap();

    ::rsp2_tasks::run_save_bands_after_the_fact(&settings, &dir).unwrap();
}
