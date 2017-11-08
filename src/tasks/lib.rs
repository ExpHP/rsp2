// HERE BE DRAGONS

extern crate rsp2_lammps_wrap;
extern crate rsp2_minimize;
extern crate rsp2_structure;
extern crate rsp2_structure_io;
extern crate rsp2_structure_gen;
extern crate rsp2_phonopy_io;
extern crate rsp2_array_utils;
extern crate rsp2_slice_math;
extern crate rsp2_tempdir;
extern crate rsp2_eigenvector_classify;
#[macro_use] extern crate rsp2_util_macros;

extern crate rayon;
extern crate rand;
extern crate slice_of_array;
extern crate serde;
extern crate ansi_term;
extern crate fern;
extern crate rsp2_kets;
extern crate serde_yaml;
#[macro_use] extern crate serde_json;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate itertools;
#[macro_use] extern crate error_chain;


macro_rules! ichain {
    ($e:expr,) => { $e.into_iter() };
    ($e:expr, $($es:expr,)+)
    => { $e.into_iter().chain(ichain!($($es,)+)) };
}

mod color;
mod util;
mod config;
mod logging;
mod cmd;
mod integrate_2d;
mod bands;

pub use ::config::Settings;
pub use ::cmd::run_relax_with_eigenvectors;
pub use ::cmd::run_symmetry_test;
pub use ::cmd::get_energy_surface;

pub use ::bands::unfold_phonon;

error_chain! {
    foreign_links {
        Io(::std::io::Error);
        Yaml(::serde_yaml::Error);
        Json(::serde_json::Error);
        SetLogger(::log::SetLoggerError);
    }

    links {
        Structure(::rsp2_structure::Error, ::rsp2_structure::ErrorKind);
        StructureIo(::rsp2_structure_io::Error, ::rsp2_structure_io::ErrorKind);
        StructureGen(::rsp2_structure_gen::Error, ::rsp2_structure_gen::ErrorKind);
        LammpsWrap(::rsp2_lammps_wrap::Error, ::rsp2_lammps_wrap::ErrorKind);
        Phonopy(::rsp2_phonopy_io::Error, ::rsp2_phonopy_io::ErrorKind);
        ExactLs(::rsp2_minimize::exact_ls::Error, ::rsp2_minimize::exact_ls::ErrorKind);
    }
}

pub type StdResult<T, E> = ::std::result::Result<T, E>;
