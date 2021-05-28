# `rsp2`

Research code for computing phonons of certain materials.

Contains a small ecosystem of utility crates for working with crystal structures.

# External dependencies

`rsp2` has the following (non-exhaustive) list of dependencies:

```
python3 >= 3.7.0 (see below about packages)
rust >= 1.35.0
libffi
libllvm and libclang
blas and lapacke
mpi (optional; see below)
clang
cmake (if a system version of lammps is not available)
```

It also has a large number of rust dependencies, but these are automatically managed by cargo.

## Python

To facilitate running the code, there is a `requirements.txt` suitable for use with `venv`.  To install all required python dependencies:

```
# First time usage
python3 -m venv venv
. venv/bin/activate
python3 -m pip install wheel  # can speed up other installs
python3 -m pip install -r requirements.txt

# Future usage
. venv/bin/activate
```

If you want to run scripts in `scripts/` (e.g. most notably for band unfolding), there are additional requirements in `scripts/requirements.txt`.

## Optional features

### MPI

MPI support is enabled by default, but you can build without MPI by supplying the `--no-default-features` flag to cargo.

**The MPI implementation must implement version 3.0 of the interface.**  (e.g. OpenMPI 3.x).  Furthermore, due to problems upstream, **openMPI 4.x is not currently supported.**  The following versions ought to work:

* OpenMPI 3.0.4, 3.1.4 (**not 4.x**)
* MPICH 3.3, 3.2.1

**MPI support is only utilized by LAMMPS potentials.**  The potentials implemented in Rust use rayon to perform work-stealing parallelism on a single machine, and the potentials in DFTB+ use OpenMP.

### LAMMPS

LAMMPS is used to provide the following potentials:

* AIREBO
* REBO + `kolmogorov/crespi/full` (with nonlocal normals; rsp2 uses local normals)
* Versions of rsp2's built-in potentials that lack support for lattice parameter optimization, but which fully model the bondorder factors in REBO.

rsp2 currently automatically builds an old version of LAMMPS.  **At the time of writing, this version is known to be incorrect for pure (non-AIREBO) REBO on some H−C bonds.**

If you want to build and install your own version of LAMMPS, [see this page](https://github.com/lammps/lammps/blob/master/cmake/README.md). To be able to use it in rsp2, please enable the following flags in the `cmake` command:

```
-DCMAKE_BUILD_TYPE=Release -DLAMMPS_EXCEPTIONS=on -DBUILD_LIB=on -DBUILD_SHARED_LIBS=on -DPKG_MANYBODY=on -DPKG_USER-OMP=on -DPKG_USER-MISC=on
```

`pkg-config` must be able to locate the library, or rsp2 will not use it.  Verify:

```
$ pkg-config --libs --cflags liblammps
-DLAMMPS_EXCEPTIONS -llammps
```

### DFTB+

`dftb+` is optional, and it is not required by default.  To enable it, supply `--features=dftbplus-support` to cargo commands.

All DFTB+ potentials are available.

You must **manually** install DFTB+. [See this page for details](https://github.com/ExpHP/dftbplus-sys/blob/master/doc/installing-dftbplus.md).

# Running

> `cargo run --release --bin=rsp2 -- --help` and *good luck*

Because there is currently one or *maybe two* people who need to use the code (and this count includes the author!), the CLI binaries have no stable interface. CLI arguments in particular may undergo major revisions on a complete whim.  Input and output file formats are a bit more stable as of late as the author has needed to work with some fairly old files, but there are no guarantees.

One or more config files is required (supplied with the `-c` flag). It will be easiest to start with an existing config file from a previous run (ask me for one, or piece one together from the pieces in `tests/`). There is, at present, no documentation of the config file that is written for end-users.  In order to work with the config file:

* The file format is YAML.
  * All mappings use `kebab-case` strings as keys.
  * There are a couple of small extensions to the format, described in the text given by `--help` for the `-c` flag.
* Documentation for the fields is provided **[in the source code](https://github.com/ExpHP/rsp2/blob/master/src/tasks/config/config.rs).**
  * Start at `struct Settings` and read the doc comments.
  * The config file itself is declaratively defined using the `serde` rust crate, so learning a bit about types in rust and how serde encodes them will be useful.

Thankfully, changes to the config file now at least generally attempt to preserve backwards compatibility.  Generally speaking, old config files will continue to work and will cause rsp2 to behave the same way as it did originally.  If a config item is renamed or relocated, you will see deprecation warnings telling you what you should write instead. (also, one of the output files from `rsp2` is a normalized `settings.yaml` file).

## MPI

You can't use `mpirun` through cargo, so you need to run the built binary directly from the target directory.

```
cargo build --release --bin=rsp2
cargo run --release --bin=rsp2-library-paths >release.path
LD_LIBRARY_PATH=$(cat release.path):${LD_LIBRARY_PATH} mpirun target/release/rsp2 ARGS GO HERE
```

## Band unfolding script

The unfolding script used to analyze the output of `rsp2` on layered 2D materials is also included in this repository.

[See this page for details](https://github.com/ExpHP/rsp2/blob/master/doc/unfolding.md).

# License

rsp2 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Many portions of rsp2 are licensed under more permissive terms (usually dual-licensed under MIT and Apache 2.0). Generally speaking, those portions which are GPL licensed are the interface to LAMMPS, and anything that uses it (transitively; hence, much of the high-level orchestration code, and the `rsp2` project as a whole).

When in doubt, consult the `Cargo.toml` files of individual subcrates.
