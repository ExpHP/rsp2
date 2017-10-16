extern crate env_logger;
extern crate sp2_tasks;
#[macro_use]
extern crate clap;
extern crate serde_yaml;

fn main() {
    let _ = ::env_logger::init();

    let matches = clap_app!(myapp =>
        (version: "negative 0.00.3-734.bubbles")
        (author: "Michael T. Lamparski")
        (about: "blah")
        (@arg OUTDIR: -o --output +takes_value +required "output directory")
        (@arg CONFIG: -c --config +takes_value +required "settings Yaml")
        (@arg INPUT: +required "POSCAR")
        (@arg force: -f --force "replace existing output directories")
        // (@subcommand test =>
        //     (about: "controls testing features")
        //     (version: "1.3")
        //     (author: "Someone E. <someone_else@other.com>")
        //     (@arg verbose: -v --verbose "Print test information verbosely")
        // )
    ).get_matches();
    let input = matches.value_of("INPUT").unwrap();
    let outdir = matches.value_of("OUTDIR").unwrap();
    let config = matches.value_of("CONFIG").unwrap();
    if matches.is_present("force") {
        ::std::fs::remove_dir_all(outdir).unwrap();
    }

    let settings = ::serde_yaml::from_reader(::std::fs::File::open(config).unwrap()).unwrap();

    let _ = ::sp2_tasks::run_relax_with_eigenvectors(&settings, input, outdir);
}