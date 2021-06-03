# Using the unfolding script

`scripts/unfold.py` is a script for generating band plots of unfolded band data.

## Worked Example

If you like to follow along with a worked example, [you can find one here](https://gist.github.com/ExpHP/d22f050b00bc345d859ecda7d1c8bf78).

## Stability warning

`unfold.py` is a monolithic tool that undergoes frequent breaking changes, especially to its CLI arguments.  If you want to write code that depends on it, you may have better luck depending on the python functions and classes that have been factored out into `unfold_lib`, as these are somewhat more stable.

## Limitations

Currently the script only works for 2D or layered materials. The plane normal should be z-oriented. The `initial.structure/meta.json` file must contain `layers` (an array with zero-based layer indices for each atom) and `layer_sc_matrices` (an array with 3x3 row-based integer coefficient matrices for each layer; the C matrices in `S = C P`, with S and P being cell matrices). Typically these must be added manually. (rsp2 will only add them if you use its exotic `layers.yaml` format for the input structure)

Note: none of these limitations apply to the stuff in `unfold_lib`.

## Usage

For instance, to unfold bands gathered from the supercell gamma point onto the primitive FBZ:

```
export PYTHONPATH=/path/to/rsp2/scripts:$PYTHONPATH
export PYTHONPATH=/path/to/rsp2/src/python:$PYTHONPATH

# Note: initial structure is used because it must have the translational symmetry
#       of the primitive structure.
unfold.py initial.structure --qpoint "0 0 0" --dynmat dynmat-gamma.npz --plot-path=GKMG --show --write-plot=plot.png
```

There is a large number of options for tweaking the plot display, including some `--plot-style` and `--plot-*-style` options for reading `mplrc` files.

The `--plot-coalesce=sum` option is highly recommended in cases where there are many (nearly-) degenerate modes with low unfold probabilities, as it will both significantly reduce SVG/PDF filesizes and will result in more accurate values of opacity.  Don't be afraid to choose a large interval size, depending on the resolution of your plot along the y axis (e.g. on a plot from 0 to 1800 wavenumber, `--plot-coalesce-threshold=4` is not unreasonable).

There are also plenty of intermediate files that can be saved and loaded to save you time on future runs. (use `--verbose` to see what's eating the most time!). Most notably, you can cache data about the eigenmodes and their unfolded probabilities to avoid having to diagonalize the dynamical matrix every time:

```
# precomputing data about eigensolutions
# (adjust --probs-threshold to trade between accuracy and filesize)
unfold.py initial.structure --qpoint "0 0 0" --dynmat dynmat-gamma.npz --write-mode-data mode-data-gamma.npz --write-probs probs-gamma.npz --probs-threshold 1e-7

# after this you no longer need --dynmat!
unfold.py initial.structure --qpoint "0 0 0" --mode-data mode-data-gamma.npz --probs probs-gamma.npz [OPTIONS...]
```

### Sampling multiple points in the supercell FBZ

For small supercells, the data produced by only looking at the supercell gamma may be pretty coarse, resulting in poor density of detail along the plot x axis.  To obtain better quality plots, you can sample multiple Q points in the supercell FBZ. (For best results, try to pick points that have many images on or near the high symmetry path of the primitive cell!).  To do this, save `mode-data` and `probs` files for each supercell Q point of interest, and write a multi-qpoint manifest file:

```
# generate dynmats at some more points in the supercell FBZ
cargo run --release --bin=rsp2-dynmat-at-q -- --qpoint "1/2 0 0" -c settings.yaml final.structure -o dynmat-m.npz
cargo run --release --bin=rsp2-dynmat-at-q -- --qpoint "1/3 1/3 0" -c settings.yaml final.structure -o dynmat-k.npz

# diagonalize those dynmats and unfold the eigensolutions
unfold.py initial.structure --qpoint "1/2 0 0" --dynmat dynmat-m.npz --write-mode-data mode-data-m.npz --write-probs probs-m.npz
unfold.py initial.structure --qpoint "1/3 1/3 0" --dynmat dynmat-k.npz --write-mode-data mode-data-k.npz --write-probs probs-k.npz

# write manifest file
cat >multi-qpoint.yaml <<HERE
- qpoint: "0 0 0"
  mode-data: "mode-data-gamma.npz"
  probs: "probs-gamma.npz"
- qpoint: "1/2 0 0"
  mode-data: "mode-data-m.npz"
  probs: "probs-m.npz"
- qpoint: "1/3 1/3 0"
  mode-data: "mode-data-k.npz"
  probs: "probs-k.npz"
HERE

# Now use this file in place of --qpoint, --dynmat, --mode-data, and/or --probs:

unfold.py initial.structure --multi-qpoint-file multi-qpoint.yaml [OPTIONS ...]
```

