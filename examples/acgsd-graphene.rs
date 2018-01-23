extern crate rsp2_tasks;
#[macro_use]
extern crate rsp2_clap;
use ::rsp2_clap::clap;

fn main() {
    use ::rsp2_tasks::relax_with_eigenvectors;
    let matches = ::clap::App::new("thing")
        .version("negative 0.00.3-734.bubbles")
        .author("Michael T. Lamparski")
        .about("blah")
        .args(&[
            arg!(*output [-o][--output]=OUTDIR "output directory"),
            arg!(*config [-c][--config]=CONFIG "settings yaml"),
            arg!( input=POSCAR "POSCAR"),
            arg!( force [-f][--force] "replace existing output directories"),
            arg!( save_bands [--save-bands] "save phonopy directory with bands at gamma"),
        ])
        .get_matches();
    let input = matches.value_of("input").unwrap();
    let outdir = matches.value_of("output").unwrap();
    let config = matches.value_of("config").unwrap();
    if matches.is_present("force") && ::std::path::Path::new(outdir).exists() {
        ::std::fs::remove_dir_all(outdir).unwrap();
    }
    let settings = ::rsp2_tasks::config::read_yaml(::std::fs::File::open(config).unwrap()).unwrap();
    let args = relax_with_eigenvectors::CliArgs {
        save_bands: matches.is_present("save_bands"),
    };
    if let Err(e) = relax_with_eigenvectors::run(&settings, &input, &outdir, args) {
        panic!("{}", e.display_chain());
    };
}


























