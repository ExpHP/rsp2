# `rsp2`

Research code for computing phonons of certain materials.

Contains a small ecosystem of utility crates for working with crystal structures.

# External dependencies

`rsp2` won't run out of the box; among the things you will need:

* `python3` with `spglib` and `scipy` installed.
* `phonopy` is an optional requirement.  (`rsp2` now reimplements most of what it needs from phonopy, but retains the ability to use phonopy directly for purposes of comparison).
* You may wish to build your own copy of LAMMPS and make it available through `pkg-config`, if you require MPI to work. See the [`lammps-sys 0.5.0` readme](https://github.com/ExpHP/lammps-sys/tree/v0.5.0).
    * If you do not need MPI support, lammps will automatically be built, but will likely be linked against the `mpi_stubs` library. This may cause incompatibilities with the rest of rsp2, which brazenly assumes that lammps was built against whatever implementation of MPI is found by `rsmpi`. In this case, for now it is recommended that you build with `--no-default-features` to disable `rsmpi` support.

If you have trouble, please open an issue.

# Running

> `cargo run --release --bin=rsp2 -- --help` and *good luck*

Because there is currently one or *maybe two* people who need to use the code (and this count includes the author!), the CLI binaries have no stable interface.  Config files, CLI arguments, inputs and outputs undergo major revisions on a complete whim.

## MPI

You can't use `mpirun` through cargo, so you need to run the built binary directly from 

```
cargo build --release --bin=rsp2
cargo run --release --bin=rsp2-library-paths >release.path
LD_LIBRARY_PATH=$(cat release.path):${LD_LIBRARY_PATH} mpirun target/release/rsp2 ARGS GO HERE
```

# License

rsp2 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Many portions of rsp2 are licensed under more permissive terms (usually dual-licensed under MIT and Apache 2.0). Generally speaking, those portions which are GPL licensed are the interface to LAMMPS, and anything that uses it (transitively; hence, much of the high-level orchestration code, and the `rsp2` project as a whole).

When in doubt, consult the `Cargo.toml` files of individual subcrates.

# Citations

TODO (for now, search the source for "Citations")
