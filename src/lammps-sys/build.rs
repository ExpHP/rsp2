extern crate bindgen;
use std::env;
use std::path::PathBuf;

fn main() {

    // Tell cargo to tell rustc to link the system lammps shared library.
    println!("cargo:rustc-link-lib=lammps");

    // Generate bindings at build time.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    ::bindgen::Builder::default()
        .header("src/wrapper.h")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
