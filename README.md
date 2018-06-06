# `rsp2`

Research code for computing phonons of certain materials.

Contains a small ecosystem of utility crates for working with crystal structures.

# External dependencies

`rsp2` won't run out of the box; among the things you will need:

* `python3` with `spglib`, `phonopy`, and `scipy` packages installed.
* You will need to build a dynamically-linked LAMMPS. See the [`lammps-sys 0.3.0` readme](https://github.com/ExpHP/lammps-sys/tree/v0.3.0).

If you have trouble, open an issue.

# Running

> `cargo run --bin=rsp2 -- --help` and *good luck*

Because there is currently exactly one person who needs to use the code (who also happens to be the author), the CLI binaries have no stable interface.  Config files, CLI arguments, inputs and outputs undergo major revisions on a whim with the needs of the author.

# License

rsp2 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Many portions of rsp2 are licensed under more permissive terms (usually dual-licensed under MIT and Apache 2.0). Generally speaking, those portions which are GPL licensed are the interface to LAMMPS, and anything that uses it (transitively; hence, much of the high-level orchestration code, and the `rsp2` project as a whole).

When in doubt, consult the `Cargo.toml` files of individual subcrates.

# Citations

TODO (for now, search the source for "Citations")
