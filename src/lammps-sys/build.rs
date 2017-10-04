fn main() {
    // Tell cargo to tell rustc to link the system lammps shared library.
    println!("cargo:rustc-link-lib=lammps");
}
