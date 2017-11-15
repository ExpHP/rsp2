extern crate rsp2_tasks;
#[macro_use]
extern crate clap;
use ::std::path::PathBuf;

fn main() {
    let matches = clap_app!(myapp =>
        (version: "negative 0.00.3-734.bubbles")
        (author: "Michael T. Lamparski")
        (about: "blah")
        (@arg POSCAR: +required "POSCAR")
        (@arg PHONOPY_CONF: "PHONOPY_CONF")
        (@arg OUTDIR: -o --output +takes_value +required "output directory")
    ).get_matches();
    let conf = matches.value_of("PHONOPY_CONF");
    let poscar = matches.value_of("POSCAR").unwrap();
    let outdir = matches.value_of("OUTDIR").unwrap();
    let conf: Option<&AsRef<::std::path::Path>> = conf.as_ref().map(|x: &_| x as _);

    ::rsp2_tasks::make_force_sets(conf, &poscar, &outdir).unwrap();
}
