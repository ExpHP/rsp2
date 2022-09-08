#!/usr/bin/env python3

import os
import numpy as np
import json
import sys
import typing as tp
from scipy import interpolate as scint
from scipy import sparse
import argparse
import itertools
from pymatgen import Structure

try:
    import rsp2
    import unfold_lib
except ImportError:
    info = lambda s: print(s, file=sys.stderr)
    info('Please add the following to your PYTHONPATH:')
    info('  (rsp2 source root)/scripts')
    info('  (rsp2 source root)/src/python')
    info("Or more preferably, create a conda environment from rsp2's environment.yml")
    sys.exit(1)

from unfold_lib import Supercell, griddata_periodic
from unfold_lib.util import check_arrays
from unfold_lib.util import map_with_progress
import unfold_lib.coalesce as coalesce

from rsp2.io import eigensols, structure_dir, dwim

# =========================================================
# NAMING CONVENTIONS:
#
# - The point in the supercell BZ describing the wavevector of
#   the computed eigenvectors is called Q, or the `qpoint`.
#   (Allen calls this K)
# - The members of the quotient space of the supercell reciprocal lattice
#   over the primitive reciprocal lattice are called G, or `gpoints`.
#   (Allen calls these G)
# - Allen assigns probabilities to each k == K + G.
#   We don't have a name for these; we simply assign probabilities to each G.
# - Points at which we are plotting are called `kpoints`.
#   (often `plot_kpoints` for clarity)
#
# SIGN CONVENTIONS:
#
# - Like the rest of rsp2, this script follows the sign convention of Phonopy.
#   The input dynmat and all internal code works under the assumption that the
#   normal mode displacements are eigenfunctions of lattice point translations
#   `T(r)` with eigenvalues `exp(-i Q.r)`.  (generally speaking, this means that
#   the displacements contain an `exp(+i Q.x)` factor for each site)
#
# - Notice this differs from Allen's paper, where translation by a lattice
#   point induces a phase of `exp(+i Q.r)`.  (in other words, each site would
#   have an `exp(-i Q.x)` factor in the displacements)
#
# =========================================================

THZ_TO_WAVENUMBER = 33.3564095198152

DEFAULT_TOL = 1e-2

A = tp.TypeVar('A')
B = tp.TypeVar('B')

def main():
    global SHOW_ACTION_STACK

    parser = argparse.ArgumentParser(
        description="Unfold phonon eigenvectors",
        epilog=
            'Uses the method of P. B. Allen et al., "Recovering hidden Bloch character: '
            'Unfolding electrons, phonons, and slabs", Phys Rev B, 87, 085322.',
        )

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('STRUCTURE', help='rsp2 structure directory')

    all_tasks = []
    def register(task):
        nonlocal all_tasks

        all_tasks.append(task)
        return task

    # Considering that all of the constructor args are explicitly type-annotated,
    # and that IntelliJ has no problem here telling you when you have the wrong
    # number of arguments, you would think that IntelliJ should be able to tell
    # you when you mix up the order of two of the arguments.
    #
    # You would be wrong.  Hence all the keyword arguments.

    structure = register(TaskStructure())

    qpoint_sfrac = register(TaskQpointSfrac())

    dynmat = register(TaskDynmat())

    eigensols = register(TaskEigensols(structure=structure, dynmat=dynmat))

    translation_deperms = register(TaskDeperms(structure=structure))

    ev_gpoint_probs = register(TaskGProbs(structure=structure, qpoint_sfrac=qpoint_sfrac, eigensols=eigensols, translation_deperms=translation_deperms))

    raw_band_path = register(TaskRawBandPath(structure=structure))

    mode_data = register(TaskEigenmodeData(eigensols=eigensols))

    raman_json = register(TaskRamanJson())

    multi_qpoint_data = register(TaskMultiQpointData(mode_data=mode_data, qpoint_sfrac=qpoint_sfrac, ev_gpoint_probs=ev_gpoint_probs))

    band_qg_indices = register(TaskRawBandQGIndices(structure=structure, multi_qpoint_data=multi_qpoint_data, raw_band_path=raw_band_path))

    band_path = register(TaskBandPath(raw_band_path=raw_band_path, band_qg_indices=band_qg_indices, multi_qpoint_data=multi_qpoint_data))

    zone_crossings = register(TaskPlotZoneCrossings(structure=structure, raw_band_path=raw_band_path))

    color_info = register(TaskPlotColorInfo(raman_json=raman_json, multi_qpoint_data=multi_qpoint_data))

    scatter_data = register(TaskBandPlotScatterData(band_path=band_path, multi_qpoint_data=multi_qpoint_data, color_info=color_info))

    _bandplot = register(TaskBandPlot(
        band_path=band_path, multi_qpoint_data=multi_qpoint_data,
        scatter_data=scatter_data, zone_crossings=zone_crossings, color_info=color_info,
    ))

    for task in all_tasks:
        task.add_parser_opts(parser)

    args = parser.parse_args()

    if not any(task.has_action(args) for task in all_tasks):
        parser.error("Nothing to do!")

    if args.debug:
        SHOW_ACTION_STACK = True

    for task in all_tasks:
        task.check_upfront(args)

    for task in all_tasks:
        if task.has_action(args):
            task.require(args)

#----------------------------------------------------------------
# CLI logic deciding when to compute certain things or e.g. to read files.
#
# Written in a vaguely declarative style with the help of a Task class
# that defers computation until it is needed.

# FIXME remove globals
ACTION_STACK = []
SHOW_ACTION_STACK = False

T = tp.TypeVar
class Task:
    NOT_YET_COMPUTED = object()

    def __init__(self):
        self.cached = Task.NOT_YET_COMPUTED

    def add_parser_opts(self, parser: argparse.ArgumentParser):
        pass

    def check_upfront(self, args):
        pass

    def has_action(self, args):
        return False

    def require(self, args):
        """ Force computation of the task, and immediately perform any actions
        associated with it (e.g. writing a file).

        It is cached after the first call so that it need not be run again.
        """
        global ACTION_STACK

        if self.cached is Task.NOT_YET_COMPUTED:
            ACTION_STACK.append(type(self).__name__)
            self.cached = self._compute(args)
            ACTION_STACK.pop()
            self._do_action(args)

        return self.cached

    def _compute(self, args):
        raise NotImplementedError

    def _do_action(self, args):
        """ A task performed after """
        pass

class TaskQpointSfrac(Task):
    def add_parser_opts(self, parser):
        parser.add_argument(
            '--qpoint', type=type(self).parse, help=
            'Q-point in fractional coordinates of the superstructure reciprocal '
            'cell, as a whitespace-separated list of 3 integers, floats, or '
            'rational numbers.',
        )

    def _compute(self, args):
        if args.qpoint is None:
            die('--qpoint is required for this action')
        return list(args.qpoint)

    @classmethod
    def parse(cls, s):
        """ Can be used by other tasks to replicate the behavior of --qpoint. """
        return parse_qpoint(s)

class TaskRamanJson(Task):
    def add_parser_opts(self, parser):
        parser.add_argument(
            '--raman-file', help=
            'rsp2 raman.json output file. Required if colorizing a plot by raman.',
        )

    def check_upfront(self, args):
        check_optional_input(args.raman_file)

    def _compute(self, args):
        if not args.raman_file:
            die('--raman-file is required for this action')

        return dwim.from_path(args.raman_file)

class TaskStructure(Task):
    def add_parser_opts(self, parser):
        # Arguments related to layer projections
        parser.add_argument(
            '--layer', type=int, default=0,
            help=
            'The output will be in the BZ of this layer (indexed from 0).',
        )

        parser.add_argument(
            '--layer-mode', choices=['one'],
            help=
            '--layer-mode=one will only consider the projection of the '
            'eigenvectors onto the primary layer, meaning the total norm of '
            'some eigenvectors in the output may be less than 1. (or even =0)',
        )

    @classmethod
    def read_dir(cls, path, layer):
        """ Can be used by other tasks to read an arbitrary path as if it were
        the STRUCTURE argument. """
        if not os.path.isdir(path):
            die('currently, only rsp2 structure directory format is supported')

        sdir = structure_dir.from_path(path)
        structure = sdir.structure
        if sdir.layer_sc_matrices is None:
            die("the structure must supply layer-sc-matrices")
        supercell = Supercell(sdir.layer_sc_matrices[layer])

        # Project everything onto a single layer
        if sdir.layers is None:
            mask = np.array([1] * len(structure))
        else:
            mask = np.array(sdir.layers) == layer

        projected_structure = Structure(
            structure.lattice,
            np.array(structure.species)[mask],
            structure.frac_coords[mask],
        )

        return {
            'supercell': supercell,
            'layer': layer,
            'mask': mask,
            'structure': structure,
            'projected_structure': projected_structure,
        }

    def _compute(self, args):
        return type(self).read_dir(args.STRUCTURE, layer=args.layer)

class TaskDeperms(Task):
    def __init__(self, structure: TaskStructure):
        super().__init__()
        self.structure = structure

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--write-perms', metavar='FILE', help=
            'Write permutations of translations to this file. (.npy, .npy.xz)',
        )

        parser.add_argument(
            '--perms', metavar='FILE', help=
            'Path to file previously written through --write-perms.',
        )

    def check_upfront(self, args):
        check_optional_input(args.perms)
        check_optional_output_ext('--write-perms', args.write_perms, forbid='.npz')

    def has_action(self, args):
        return bool(args.write_perms)

    def _compute(self, args):
        if args.perms:
            return np.array(dwim.from_path(args.perms))
        else:
            progress_callback = None
            if args.verbose:
                def progress_callback(done, count):
                    end = '\r' if sys.stdout.isatty() else None
                    print(f'Deperms: {done:>5} of {count} translations', end=end)

            out = collect_translation_deperms(
                superstructure=self.structure.require(args)['projected_structure'],
                supercell=self.structure.require(args)['supercell'],
                axis_mask=np.array([1,1,0]),
                tol=DEFAULT_TOL,
                progress=progress_callback,
            )
            if sys.stdout.isatty():
                print()
            return out

    def _do_action(self, args):
        if args.write_perms:
            translation_deperms = self.require(args)
            dwim.to_path(args.write_perms, translation_deperms)

class TaskDynmat(Task):
    def add_parser_opts(self, parser):
        parser.add_argument(
            '--dynmat', metavar='FILE', help='rsp2 dynmat file (.npz)',
        )

    def check_upfront(self, args):
        check_optional_input(args.dynmat)

    def _compute(self, args):
        if not args.dynmat:
            die('--dynmat is required for this action')

        m = sparse.load_npz(args.dynmat)
        if np.all(m.data.imag == 0.0):
            m = m.real
        return m

class TaskEigensols(Task):
    def __init__(self, structure: TaskStructure, dynmat: TaskDynmat):
        super().__init__()
        self.structure = structure
        self.dynmat = dynmat

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--eigensols', metavar='FILE',
            help='read rsp2 eigensols file. (.npz)',
        )

        parser.add_argument(
            '--write-eigensols', metavar='FILE',
            help='write rsp2 eigensols file. (.npz)',
        )

    def check_upfront(self, args):
        check_optional_input(args.eigensols)
        check_optional_output_ext('--write-eigensols', args.write_eigensols, forbid='.npy')

    def has_action(self, args):
        return bool(args.write_eigensols)

    def _compute(self, args):
        mask = self.structure.require(args)['mask']
        nsites = len(mask)

        if args.eigensols:
            if args.verbose:
                # This file can be very large and reading it can take a long time
                print('Reading eigensols file')

            ev_eigenvalues, ev_eigenvectors = eigensols.from_path(args.eigensols)

        else:
            if args.verbose:
                print('--eigensols not supplied. Will diagonalize dynmat.')

            m = self.dynmat.require(args)
            ev_eigenvalues, ev_eigenvectors = unfold_lib.destructive_eigh(m.todense())
            ev_eigenvectors = ev_eigenvectors.T

        # use less memory in the common case where atoms are ordered by layer
        mask_or_slice = to_slice_if_possible(mask)

        ev_projected_eigenvectors = ev_eigenvectors.reshape((-1, nsites, 3))[:, mask_or_slice]

        return {
            'ev_eigenvalues': ev_eigenvalues,
            'ev_eigenvectors': ev_eigenvectors,
            'ev_projected_eigenvectors': ev_projected_eigenvectors,
        }

    def _do_action(self, args):
        if args.write_eigensols:
            d = self.require(args)
            esols = d['ev_eigenvalues'], d['ev_eigenvectors']
            eigensols.to_path(args.write_eigensols, esols)

class TaskEigenmodeData(Task):
    """ Scalar data about eigenmodes for the plot. """
    def __init__(self, eigensols: TaskEigensols):
        super().__init__()
        self.eigensols = eigensols

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--write-mode-data', metavar='FILE', help=
            'Write data about plotted eigenmodes to this file. (.npz)',
        )

        parser.add_argument(
            '--mode-data', metavar='FILE', help=
            'Read data previously written using --write-mode-data so that reading '
            'the (large) eigensols file is not necessary to produce a plot.',
        )

    def check_upfront(self, args):
        check_optional_input(args.mode_data)
        check_optional_output_ext('--write-mode-data', args.write_mode_data, forbid='.npy')

    def has_action(self, args):
        return bool(args.write_mode_data)

    def _compute(self, args):
        if args.mode_data:
            return type(self).read_file(args.mode_data)

        if args.verbose:
            print('--mode-data not supplied. Computing from eigensols.')

        ev_eigenvalues = self.eigensols.require(args)['ev_eigenvalues']
        ev_eigenvectors = self.eigensols.require(args)['ev_eigenvectors']

        ev_frequencies = eigensols.wavenumber_from_eigenvalue(ev_eigenvalues)

        ev_z_coords = ev_eigenvectors.reshape((-1, ev_eigenvectors.shape[1]//3, 3))[:, :, 2]
        ev_z_projections = np.linalg.norm(ev_z_coords, axis=1)**2
        return {
            'ev_frequencies': ev_frequencies,
            'ev_z_projections': ev_z_projections,
        }

    def _do_action(self, args):
        if args.write_mode_data:
            d = self.require(args)
            np.savez_compressed(
                args.write_mode_data,
                ev_frequencies=d['ev_frequencies'],
                ev_z_projections=d['ev_z_projections'],
            )

    @classmethod
    def read_file(cls, path):
        """ Can be used by other tasks to replicate the behavior of --mode-data. """
        npz = np.load(path)
        return {
            'ev_frequencies': npz.f.ev_frequencies,
            'ev_z_projections': npz.f.ev_z_projections,
        }

# Arguments related to probabilities
# (an intermediate file format that can be significantly smaller than the
#  input eigenvectors and thus easier to transmit)
class TaskGProbs(Task):
    def __init__(
            self,
            structure: TaskStructure,
            qpoint_sfrac: TaskQpointSfrac,
            eigensols: TaskEigensols,
            translation_deperms: TaskDeperms,
    ):
        super().__init__()
        self.structure = structure
        self.qpoint_sfrac = qpoint_sfrac
        self.eigensols = eigensols
        self.translation_deperms = translation_deperms

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--write-probs', metavar='FILE', help=
            'Write magnitudes of g-point projections to this file. (.npz)',
        )

        parser.add_argument(
            '--probs-threshold', type=float, default=1e-7, help=
            'Truncate probabilities smaller than this when writing probs. '
            'This can significantly reduce disk usage.',
        )

        parser.add_argument(
            '--probs-impl', choices=['auto', 'rust', 'python'], default='auto', help=
            'Enable the experimental rust unfolder.',
        )

        parser.add_argument(
            '--probs', metavar='FILE', help=
            'Path to .npz file previously written through --write-probs.',
        )

        parser.add_argument(
            '--probs-gamma-only', action='store_true', help=
            'Only compute the probability at gamma.  The probs file will report '
            'zero probability for all other projections.  Ignored when reading '
            'a probs file, and forbidden when computing things that require all '
            'probabilities. (e.g. plotting)',
        )

    def check_upfront(self, args: argparse.Namespace):
        check_optional_input(args.probs)
        check_optional_output_ext('--write-probs', args.write_probs, forbid='.npy')

        if args.probs_gamma_only and self.qpoint_sfrac.require(args) != [0, 0, 0]:
            raise RuntimeError('--probs-gamma-only requires --qpoint "0 0 0"')

        unfold_lib.prepare_implementation(args.probs_impl)

    def has_action(self, args: argparse.Namespace):
        return bool(args.write_probs)

    def _compute(self, args: argparse.Namespace):
        if args.probs:
            ev_gpoint_probs = type(self).read_file(
                args.probs,
                args.probs_threshold,
                verbose=args.verbose,
            )
        else:
            if args.verbose:
                print('--probs not supplied. Will compute by unfolding eigensols.')

            layer = self.structure.require(args)['layer']

            progress_prefix = f'Layer {layer}: ' if args.verbose else None

            # reading the file might take forever; compute deperms first as it has
            # a greater chance of having a bug
            self.translation_deperms.require(args)

            ev_gpoint_probs = unfold_lib.unfold_all(
                superstructure=self.structure.require(args)['projected_structure'],
                supercell=self.structure.require(args)['supercell'],
                eigenvectors=self.eigensols.require(args)['ev_projected_eigenvectors'],
                qpoint_sfrac=self.qpoint_sfrac.require(args),
                translation_deperms=self.translation_deperms.require(args),
                gamma_only=args.probs_gamma_only,
                implementation=args.probs_impl,
                progress_prefix=progress_prefix,
            )
        ev_gpoint_probs = type(self).__postprocess(
            ev_gpoint_probs,
            probs_threshold=args.probs_threshold,
            verbose=args.verbose,
        )

        if args.probs_gamma_only:
            return { 'raw': ev_gpoint_probs }
        else:
            return { 'raw': ev_gpoint_probs, 'full': ev_gpoint_probs }

    def require_full(self, args: argparse.Namespace):
        if args.probs_gamma_only:
            die('--probs-gamma-only is incompatible with some of the requested actions')
        return self.require(args)['full']

    def _do_action(self, args: argparse.Namespace):
        ev_gpoint_probs = self.require(args)['raw']
        if args.write_probs:
            dwim.to_path(args.write_probs, ev_gpoint_probs)

    @classmethod
    def read_file(cls, path: str, probs_threshold: float, verbose: bool = False):
        """ Can be used by other tasks to replicate the behavior of --probs. """
        ev_gpoint_probs = dwim.from_path(path)
        return cls.__postprocess(ev_gpoint_probs, verbose=verbose, probs_threshold=probs_threshold)

    @classmethod
    def __postprocess(cls, ev_gpoint_probs, probs_threshold, verbose):
        if verbose:
            debug_bin_magnitudes(ev_gpoint_probs)

        ev_gpoint_probs = truncate(ev_gpoint_probs, probs_threshold)
        ev_gpoint_probs = sparse.csr_matrix(ev_gpoint_probs)

        if verbose:
            density = ev_gpoint_probs.nnz / product(ev_gpoint_probs.shape)
            print('Probs matrix density: {:.4g}%'.format(100.0 * density))

        return ev_gpoint_probs

class TaskRawBandPath(Task):
    def __init__(
            self,
            structure: TaskStructure,
    ):
        super().__init__()
        self.structure = structure

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--plot-path', dest='plot_kpath_str', help=
            "A kpath in the format accepted by ASE's parse_path_string, "
            "naming points in the monolayer BZ.  If not specified, no band "
            "plot is generated."
        )

        # I'm thinking of adding back the --bands file, in which case this
        # argument name makes more sense. (I removed it because it is super
        # quick to generate; I might want it back because it could be much
        # smaller than the input it is generated from!)
        parser.add_argument(
            '--band-path', dest='plot_kpath_str', help=
            "Alias for --plot-path."
        )

        parser.add_argument(
            '--write-highsym-xs', metavar='FILE', help=
            'Record x locations of high symmetry points. If --write-scatter-data '
            'is used, the values written to this file will be exactly equal to '
            'some of the x values in that file.  May be any format supported by '
            "rsp2's dwim IO facilities (e.g. .json.xz).",
        )

    def has_action(self, args):
        return bool(args.write_highsym_xs)

    def _compute(self, args):
        from ase.dft.kpoints import bandpath

        supercell = self.structure.require(args)['supercell']
        super_lattice = self.structure.require(args)['structure'].lattice.matrix

        prim_lattice = np.linalg.inv(supercell.matrix) @ super_lattice

        if args.plot_kpath_str is None:
            die('--plot-path is required')

        # NOTE: The kpoints returned by get_special_points (and by proxy, this
        #       function) do adapt to the user's specific choice of primitive cell.
        #       (at least, for reasonable cells; I haven't tested it with a highly
        #       skewed cell). Respect!
        path = bandpath(args.plot_kpath_str, prim_lattice, 300)
        path_kpoint_pfracs = path.kpts
        path_x_coordinates, plot_xticks, point_names = path.get_linear_kpoint_axis()
        point_names = [r'$\mathrm{\Gamma}$' if x == 'G' else x for x in point_names]

        highsym_pfracs = bandpath(args.plot_kpath_str, prim_lattice, 1).get_linear_kpoint_axis()[0]

        return {
            'path_kpoint_pfracs': path_kpoint_pfracs,
            'path_x_coordinates': path_x_coordinates,
            'plot_xticks': plot_xticks,
            'plot_xticklabels': point_names,
            'highsym_pfracs': highsym_pfracs,
        }

    def _do_action(self, args):
        if args.write_highsym_xs:
            dwim.to_path(args.write_highsym_xs, self.require(args)['plot_xticks'].tolist())

class TaskMultiQpointData(Task):
    def __init__(
            self,
            mode_data: TaskEigenmodeData,
            qpoint_sfrac: TaskQpointSfrac,
            ev_gpoint_probs: TaskGProbs,
    ):
        super().__init__()
        self.mode_data = mode_data
        self.qpoint_sfrac = qpoint_sfrac
        self.ev_gpoint_probs = ev_gpoint_probs

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--multi-qpoint-file', help=
            "Multi-qpoint manifest file.  This allows using data from multiple "
            "qpoints to be included on a single plot. If this is supplied, many "
            "arguments for dealing with a single qpoint (e.g. --dynmat, --qpoint) "
            f"will be ignored.\n\n{MULTI_QPOINT_FILE_HELP_STR}"
        )

    def check_upfront(self, args):
        check_optional_input(args.multi_qpoint_file)

    def _compute(self, args):
        if args.multi_qpoint_file:
            return type(self).read_file(
                args.multi_qpoint_file,
                verbose=args.verbose,
                probs_threshold=args.probs_threshold,
            )
        else:
            return type(self).__process_dicts({
                "qpoint-sfrac": self.qpoint_sfrac.require(args),
                "mode-data": self.mode_data.require(args),
                "probs": self.ev_gpoint_probs.require_full(args),
            })

    @classmethod
    def read_file(cls, path, probs_threshold, verbose=False):
        d = dwim.from_path(path)
        if not isinstance(d, list):
            die(f'Expected {path} to contain a sequence/array.')

        base_dir = os.path.dirname(path)
        rel_path = lambda name: os.path.join(base_dir, name)

        dicts = []
        unrecognized_keys = set()
        for item in d:
            dicts.append({
                "qpoint-sfrac": TaskQpointSfrac.parse(item.pop('qpoint')),
                "mode-data": TaskEigenmodeData.read_file(rel_path(item.pop('mode-data'))),
                "probs": TaskGProbs.read_file(rel_path(item.pop('probs')), probs_threshold=probs_threshold,  verbose=verbose),
            })
            unrecognized_keys.update(item)

        if unrecognized_keys:
            warn(f"Unrecognized keys in multi-qpoint manifest: {repr(sorted(unrecognized_keys))}")

        return cls.__process_dicts(*dicts)

    @classmethod
    def __process_dicts(cls, *dicts):
        dict_of_lists = dict_zip(*dicts)
        dict_of_lists['mode-data'] = dict_zip(*dict_of_lists['mode-data'])
        dict_of_lists['num-qpoints'] = len(dict_of_lists['qpoint-sfrac'])
        return dict_of_lists

MULTI_QPOINT_FILE_KEYS = ["qpoint", "probs", "mode-data"]

MULTI_QPOINT_FILE_HELP_STR = f"""
The multi-qpoint manifest is a sequence (encoded in JSON or YAML) whose elements
are mappings with the keys: {repr(MULTI_QPOINT_FILE_KEYS)}. Each of these keys
maps to a string exactly like the corresponding CLI argument.  This means that
in order to use this option, you will first need to generate files at each
Q-point in individual runs using --write-probs and --write-mode-data.
""".strip().replace('\n', ' ')

# Performs resampling along the high symmetry path.
class TaskRawBandQGIndices(Task):
    def __init__(
            self,
            structure: TaskStructure,
            raw_band_path: TaskRawBandPath,
            multi_qpoint_data: TaskMultiQpointData,
    ):
        super().__init__()
        self.structure = structure
        self.raw_band_path = raw_band_path
        self.multi_qpoint_data = multi_qpoint_data

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--write-debug-path-plot', help=
            'Write a plot displaying both the bandplot path and full set of points in the primitive reciprocal cell '
            'that have unfolded probabilities assigned to them.  This can help identify scenarios in small supercells '
            'where there are large regions of the band path that do not have any nearby points with data (which can '
            'create discontinuities in the band plot). '
            "You can use the path '-' to show the plot via plt.show()."
        )

    def has_action(self, args):
        return bool(args.write_debug_path_plot)

    def _compute(self, args):
        return resample_qg_indices(
                super_lattice=self.structure.require(args)['structure'].lattice.matrix,
                supercell=self.structure.require(args)['supercell'],
                path_kpoint_pfracs=self.raw_band_path.require(args)['path_kpoint_pfracs'],
                qpoint_sfrac=self.multi_qpoint_data.require(args)['qpoint-sfrac'],
                debug_path_outpath=args.write_debug_path_plot,
        )

class TaskPlotZoneCrossings(Task):
    def __init__(
            self,
            structure: TaskStructure,
            raw_band_path: TaskRawBandPath,
    ):
        super().__init__()
        self.structure = structure
        self.raw_band_path = raw_band_path

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--plot-zone-crossings', choices=['parallel', 'voronoi'], help=
            'Draw lines at each zone crossing.'
        )

    def check_upfront(self, args):
        if args.plot_zone_crossings == 'voronoi':
            die('--plot-zone-crossings=voronoi is not implemented')

    def _compute(self, args):
        if args.plot_zone_crossings:
            assert args.plot_zone_crossings == 'parallel'
            zone_crossing_xs = get_parallelogram_zone_crossings(
                highsym_pfracs=self.raw_band_path.require(args)['highsym_pfracs'],
                supercell=self.structure.require(args)['supercell'],
                plot_xticks=self.raw_band_path.require(args)['plot_xticks'],
            )
        else:
            zone_crossing_xs = np.array([])

        return { "xs": zone_crossing_xs }

class TaskPlotColorInfo(Task):
    def __init__(
            self,
            raman_json: TaskRamanJson,
            multi_qpoint_data: TaskMultiQpointData,
    ):
        super().__init__()
        self.raman_json = raman_json
        self.multi_qpoint_data = multi_qpoint_data

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--plot-color', type=str, default='zpol', metavar='SCHEME', help=
            'How the plot points are colored. Choices: [zpol, uniform:COLOR, raman:POL, prob:CMAP] '
            '(e.g. --plot-color uniform:blue). POL is either "average-3d" or "backscatter". '
            'CMAP names a matplotlib colormap.'
        )

    def _compute(self, args):
        cfg_matplotlib()

        multi_qpoint_data = self.multi_qpoint_data.require(args)

        raman_dict = None
        if args.plot_color.startswith('raman:'):
            if multi_qpoint_data['num-qpoints'] == 1:
                # make the dict items indexed by [qpoint (just the one)][ev]
                raman_dict = self.raman_json.require(args)
                raman_dict = { k: np.array([v]) for (k, v) in raman_dict.items() }
            else:
                warn('raman coloring cannot be used with multiple kpoints')
                args.plot_color = 'zpol'

        mode_data = multi_qpoint_data['mode-data']
        q_ev_z_projections = np.array(mode_data['ev_z_projections'])

        color_info = get_plot_color_info(args.plot_color, z_pol=q_ev_z_projections, raman_dict=raman_dict)

        if args.plot_colorbar and color_info.cbar_info() is None:
            die("--plot-colorbar doesn't make sense for the given --plot-color mode")
            raise RuntimeError('unreachable')

        return color_info

class TaskBandPath(Task):
    def __init__(
            self,
            raw_band_path: TaskRawBandPath,
            band_qg_indices: TaskRawBandQGIndices,
            multi_qpoint_data: TaskMultiQpointData,
    ):
        super().__init__()
        self.raw_band_path = raw_band_path
        self.band_qg_indices = band_qg_indices
        self.multi_qpoint_data = multi_qpoint_data

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--decimate-x', type=int, metavar='N', default=1, help=
            'When multiple consecutive x points on the plot map to the same '
            '(Q, G) indices, only include every Nth point. The points that '
            'land on top of the high symmetry points are always included.'
        )

    def _compute(self, args):
        return decimate_plot_x(
            {
                "highsym_pfracs": self.raw_band_path.require(args)['highsym_pfracs'],
                "path_kpoint_pfracs": self.raw_band_path.require(args)['path_kpoint_pfracs'],
                "path_x_coordinates": self.raw_band_path.require(args)['path_x_coordinates'],
                "plot_xticks": self.raw_band_path.require(args)['plot_xticks'],
                "plot_xticklabels": self.raw_band_path.require(args)['plot_xticklabels'],
                "path_q_indices": self.band_qg_indices.require(args)['Q'],
                "path_g_indices": self.band_qg_indices.require(args)['G'],
            },
            decimate_x=args.decimate_x,
        )

class TaskBandPlotScatterData(Task):
    def __init__(
            self,
            band_path: TaskBandPath,
            multi_qpoint_data: TaskMultiQpointData,
            color_info: TaskPlotColorInfo,
    ):
        super().__init__()
        self.band_path = band_path
        self.multi_qpoint_data = multi_qpoint_data
        self.color_info = color_info

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--write-scatter-data', metavar='FILE', help=
            'Write data for a scatter plot, containing color data and total '
            'probability (after coalescing similar modes) for each point. '
            "Output is any format supported by rsp2's dwim IO facilities "
            '(e.g. .json.gz), and will contain a mapping of string keys '
            '(e.g. "x") to lists of floats.'
        )

        parser.add_argument(
            '--scatter-data', metavar='FILE', help=
            'Path to file previously written through --write-scatter-data, '
            'allowing one to skip the somewhat expensive step of coalescing.'
        )

        # FIXME: Remove 'max' option.
        parser.add_argument(
            '--plot-coalesce', choices=['none', 'max', 'sum'], default='none',
            metavar='MODE', help=
            'Coalesce modes of similar eigenvalue into one. '
            'The argument decides how to treat the probabilities.'
        )

        parser.add_argument(
            '--plot-coalesce-threshold', type=float, default=0.1,
            metavar='THRESHOLD', help=
            'Threshold used by --plot-coalesce. (cm^-1)'
        )

        parser.add_argument(
            '--plot-truncate-coalesced', type=float, metavar='VALUE', default=0.0, help=
            'Eliminate points whose coalesced probability is less than this. '
            'In contrast to --plot-truncate, this option will ignore the '
            'effects of --plot-exponent and --plot-max-alpha, and it will be '
            'reflected in the output of --write-scatter-data. '
            "Prefer to use --plot-truncate unless using --write-scatter-data."
        )

    def check_upfront(self, args):
        check_optional_input(args.scatter_data)
        check_optional_output_ext('--write-probs', args.write_scatter_data, forbid=['.npy', '.npz'])

    def has_action(self, args):
        return bool(args.write_scatter_data)

    def _compute(self, args):
        cfg_matplotlib()

        if args.scatter_data:
            return type(self).read_file(args.scatter_data)

        # Do this early so we can validate it before doing anything expensive.
        band_path = self.band_path.require(args)
        color_info = self.color_info.require(args) # note: also requires multi_qpoint_data

        multi_qpoint_data = self.multi_qpoint_data.require(args)

        return compute_band_plot_scatter_data(
            q_ev_frequencies=np.array(multi_qpoint_data['mode-data']['ev_frequencies']),
            q_ev_gpoint_probs=np.array(multi_qpoint_data['probs']),
            path_g_indices=band_path['path_g_indices'],
            path_q_indices=band_path['path_q_indices'],
            path_x_coordinates=band_path['path_x_coordinates'],
            color_info=color_info,
            plot_coalesce_method=args.plot_coalesce,
            plot_coalesce_threshold=args.plot_coalesce_threshold,
            coalesced_truncate=args.plot_truncate_coalesced,
            verbose=args.verbose,
        )

    @classmethod
    def read_file(cls, path):
        return {
            key: np.array(lst)
            for (key, lst) in dwim.from_path(path).items()
        }

    def _do_action(self, args):
        if args.write_scatter_data:
            d = self.require(args)
            d = { key: array.tolist() for (key, array) in d.items() }
            dwim.to_path(args.write_scatter_data, d)

class TaskBandPlot(Task):
    def __init__(
            self,
            band_path: TaskBandPath,
            scatter_data: TaskBandPlotScatterData,
            multi_qpoint_data: TaskMultiQpointData,
            zone_crossings: TaskPlotZoneCrossings,
            color_info: TaskPlotColorInfo,
    ):
        super().__init__()
        self.band_path = band_path
        self.scatter_data = scatter_data
        self.multi_qpoint_data = multi_qpoint_data
        self.zone_crossings = zone_crossings
        self.color_info = color_info

    def add_parser_opts(self, parser):
        parser.add_argument('--show', action='store_true', help='show plot')
        parser.add_argument(
            '--write-plot', metavar='FILE', action='append', default=[], help=
            'Save plot to file. Can be of any file type supported by matplotlib. '
            'May be specified multiple times, in case you want multiple formats.',
        )

        parser.add_argument(
            '--plot-baseline-file', type=str, metavar='FILE', help=
            'Data file for a "normal" plot. (e.g. bands from the structure whose BZ '
            'we are unfolding into).  Phonopy band.yaml is accepted.'
        )

        parser.add_argument(
            '--plot-sidebar', action='store_true', help=
            'Show a sidebar with the frequencies all on the same point.'
        )

        parser.add_argument(
            '--plot-colorbar', action='store_true', help=
            'Show a colorbar.'
        )

        parser.add_argument(
            '--plot-size', type=parse_figsize, default=(7, 8), help=
            'Set figure size.'
        )

        parser.add_argument(
            '--plot-ylim', type=parse_ylim, default=(None, None), help=
            'Set plot ylim.'
        )

        parser.add_argument(
            '--plot-style', metavar='STYLE', action='append', default=[], help=
            'Apply a matplotlib stylesheet to the entire plot. '
            'This can be supplied multiple times.'
        )

        # because axes.prop_cycle in .mplrc files doesn't seem to work
        parser.add_argument(
            '--plot-baseline-color', metavar='COLOR', default='uniform:#770000', help=
            'Set color for the baseline.  Must be of form "uniform:MPLCOLORSTRING"'
        )

        parser.add_argument(
            '--plot-unfolded-style', metavar='STYLE', action='append', default=[], help=
            'Apply a matplotlib stylesheet to the scatter plot for the '
            'unfolded bands. This can be supplied multiple times.'
        )

        parser.add_argument(
            '--plot-baseline-style', metavar='STYLE', action='append', default=[], help=
            'Apply a matplotlib stylesheet to the baseline scatter plot if '
            'there is one. This can be supplied multiple times.'
        )

        parser.add_argument(
            '--plot-baseline-mode', metavar='MODE', default='scatter', choices=['scatter', 'line'], help=
            "'scatter' or 'line' for plotting the baseline bands."
        )

        parser.add_argument(
            '--plot-title', metavar='TITLE', help=
            'Title of plot.'
        )

        parser.add_argument(
            '--plot-hide-unfolded', action='store_true', help=
            "Don't actually show the unfolded probs. (intended for use with --plot-baseline-file, "
            "so that you can show only the baseline)"
        )

        parser.add_argument(
            '--plot-using-size', metavar='MODE',
            nargs='?', const='only', choices=['both', 'only'], help=
            'Use marker size instead of alpha to represent probability. '
            '--plot-using-size=both will use size AND alpha.'
        )

        parser.add_argument(
            '--plot-exponent', type=float, metavar='VALUE', default=1.0, help=
            'Scale probabilities by this exponent before plotting.'
        )

        parser.add_argument(
            '--plot-max-alpha', type=float, metavar='VALUE', default=1.0, help=
            'Scale probabilities by this factor before plotting.'
        )

        parser.add_argument(
            '--plot-scatter-size', type=float, metavar='SIZE', default=20, help=
            'Maximum scatter point size.'
        )

        parser.add_argument(
            '--plot-truncate', type=float, metavar='VALUE', default=1e-3, help=
            'Don\'t plot points whose final alpha is less than this. '
            'This can be a good idea for SVG and PDF outputs.'
        )

        parser.add_argument(
            '--plot-dpi', type=int, metavar='DPI', default=None, help=
            'DPI when saving e.g. a PNG image using --write-plot.'
        )

    def check_upfront(self, args):
        check_optional_input(args.plot_baseline_file)

        for path in itertools.chain.from_iterable([
            args.plot_style,
            args.plot_unfolded_style,
            args.plot_baseline_style,
        ]):
            try:
                # Allow MPL to check globally installed styles in addition to
                # absolute/relative filepaths.
                with mpl_context(path):
                    pass
            except OSError:
                # Probably a missing file, in which case this should fail
                # with a nice error message.
                check_optional_input(path)

                # but in case that didn't fail...
                raise

    def has_action(self, args):
        return args.show or bool(args.write_plot)

    def _compute(self, args):
        cfg_matplotlib()

        # Do this early so we can validate it before doing anything expensive.
        color_info = self.color_info.require(args)

        multi_qpoint_data = self.multi_qpoint_data.require(args)

        if args.plot_sidebar and len(multi_qpoint_data) > 1:
            warn("--plot-sidebar doesn't make sense with multiple kpoints")

        if args.plot_baseline_file is not None:
            base_X, base_Y = read_baseline_plot(args.plot_baseline_file)
            baseline_data = { 'X': np.array(base_X), 'Y': np.array(base_Y) }
        else:
            baseline_data = { 'X': np.array([]), 'Y': np.array([]) }

        scatter_data = self.scatter_data.require(args)

        Alpha = scatter_data['prob'].copy()
        Alpha **= args.plot_exponent
        Alpha *= args.plot_max_alpha
        Alpha = np.minimum(Alpha, 1)
        mask = Alpha > args.plot_truncate

        scatter_data['alpha'] = Alpha
        scatter_data = { key: array[mask] for (key, array) in scatter_data.items() }

        if not args.plot_baseline_color.startswith('uniform:'):
            die(f'invalid baseline color string: {repr(args.plot_baseline_color)} (must begin with "uniform:")')
        plot_baseline_color = args.plot_baseline_color[len('uniform:'):]

        return generate_band_plot(
            scatter_data=scatter_data,
            baseline_data=baseline_data,
            color_info=color_info,
            plot_style=args.plot_style,
            plot_unfolded_style=args.plot_unfolded_style,
            plot_baseline_style=args.plot_baseline_style,
            plot_baseline_mode=args.plot_baseline_mode,
            plot_baseline_color=plot_baseline_color,
            plot_xmax=self.band_path.require(args)['path_x_coordinates'][-1],
            plot_xticks=self.band_path.require(args)['plot_xticks'],
            plot_xticklabels=self.band_path.require(args)['plot_xticklabels'],
            plot_ylim=args.plot_ylim,
            plot_zone_crossing_xs=self.zone_crossings.require(args)['xs'],
            plot_baseline_path=args.plot_baseline_file,
            plot_title=args.plot_title,
            plot_sidebar=args.plot_sidebar,
            plot_colorbar=args.plot_colorbar,
            plot_hide_unfolded=args.plot_hide_unfolded,
            plot_using_size=args.plot_using_size,
            plot_scatter_size=args.plot_scatter_size,
            verbose=args.verbose,
        )

    def _do_action(self, args):
        fig, ax = self.require(args)

        for path in args.write_plot:
            kw = {}
            if args.plot_dpi is not None:
                kw['dpi'] = args.plot_dpi
            if args.verbose:
                print(f'Writing {path}')
            fig.savefig(path, **kw)

        if args.show:
            if args.verbose:
                print(f'Rendering preview')
            import matplotlib.pyplot as plt
            # fig.show() # doesn't do anything :/
            plt.show()

#---------------------------------------------------------------
# Computation

def collect_translation_deperms(
        superstructure: Structure,
        supercell: 'Supercell',
        axis_mask = np.array([True, True, True]),
        tol: float = DEFAULT_TOL,
        progress = None,
):
    """
    :param superstructure:
    :param supercell:

    :param axis_mask: Shape ``(3,)``, boolean.
    Permutation finding can be troublesome if the primitive cell translational
    symmetry is very strongly broken along some axis (e.g. formation of ripples
    in a sheet of graphene).  This can be used to filter those axes out of these
    permutation searches.

    :param tol: ``float``.
    Cartesian distance within which sites must overlap to be considered
    equivalent.

    :param progress: Progress callback.
    Called as ``progress(num_done, num_total)``.
    :return:
    """
    super_lattice = superstructure.lattice.matrix
    prim_lattice = np.linalg.inv(supercell.matrix) @ super_lattice

    # The quotient group of primitive lattice translations modulo the supercell
    translation_carts = supercell.translation_pfracs() @ prim_lattice

    # debug_quotient_points(translation_carts[:, :2], super_lattice[:2,:2])

    # Allen's method requires the supercell to approximately resemble the
    # primitive cell, so that translations of the eigenvector can be emulated
    # by permuting its data.
    return list(map_with_progress(
        translation_carts, progress,
        lambda translation_cart: get_translation_deperm(
            structure=superstructure,
            translation_cart=translation_cart,
            axis_mask=axis_mask,
            tol=tol,
        ),
    ))

def get_translation_deperm(
        structure: Structure,
        translation_cart,
        axis_mask = np.array([1, 1, 1]),
        tol: float = DEFAULT_TOL,
):
    lattice = structure.lattice.matrix
    fracs_original = structure.frac_coords
    fracs_translated = (structure.cart_coords + translation_cart) @ np.linalg.inv(lattice)

    # Possibly ignore displacements along certain axes.
    fracs_original *= axis_mask
    fracs_translated *= axis_mask

    return unfold_lib.compute_depermutation(
        fracs_original, fracs_translated, lattice, tol,
    )

# Here we encounter a big problem:
#
#     The points at which we want to draw bands are not necessarily
#     images of the qpoint at which we computed eigenvectors.
#
# Our solution is not very rigorous. For each point on the plot's x-axis, we
# will simply produce the projected probabilities onto the nearest image of
# the supercell qpoint.
#
# The idea is that for large supercells, every point in the primitive BZ
# is close to an image of the supercell qpoint point. (though this scheme
# may fail to produce certain physical effects that are specifically enabled
# by the symmetry of a high-symmetry point when that point is not an image
# of the qpoint)
#
# With the addition of --multi-qpoint-file, the density of the points we
# are sampling from can be even further increased.
def resample_qg_indices(
        super_lattice,
        supercell,
        qpoint_sfrac,
        path_kpoint_pfracs,
        debug_path_outpath: tp.Optional[str],
):
    gpoint_sfracs = supercell.gpoint_sfracs()

    sizes = check_arrays(
        super_lattice = (super_lattice, [3, 3], np.floating),
        gpoint_sfracs = (gpoint_sfracs, ['quotient', 3], np.floating),
        qpoint_sfrac = (qpoint_sfrac, ['qpoint', 3], np.floating),
        path_kpoint_pfracs = (path_kpoint_pfracs, ['plot-x', 3], np.floating),
    )

    prim_lattice = np.linalg.inv(supercell.matrix) @ super_lattice
    prim_recip_lattice = np.linalg.inv(prim_lattice).T

    # All of the (Q + G) points at which probabilities were computed.
    qg_sfracs = np.vstack([
        supercell.gpoint_sfracs() + sfrac
        for sfrac in qpoint_sfrac
    ])
    qg_carts = qg_sfracs @ np.linalg.inv(super_lattice).T
    assert qg_carts.shape == (sizes['qpoint'] * sizes['quotient'], 3)

    # For each of those (Q + G) points, the index of its Q point and its K point.
    qg_q_ids, qg_g_ids = np.mgrid[0:sizes['qpoint'], 0:sizes['quotient']].reshape((2, -1))

    # For every point on the plot x-axis, the index of the closest Q + G point
    plot_kpoint_carts = path_kpoint_pfracs @ prim_recip_lattice
    plot_kpoint_qg_ids = griddata_periodic(
        points=qg_carts,
        values=np.arange(sizes['qpoint'] * sizes['quotient']),
        xi=plot_kpoint_carts,
        lattice=prim_recip_lattice,
        periodic_axis_mask=[1,1,0],
        method='nearest',
        _supercell=supercell,
    )

    if debug_path_outpath:
        fig, _ax = debug_path(
            points=qg_carts[:, :2],
            lattice=prim_recip_lattice[:2, :2],
            path=plot_kpoint_carts[:, :2],
            supercell=supercell.recip(),
        )
        if debug_path_outpath == '-':
            import matplotlib.pyplot as plt
            plt.show()
        else:
            fig.savefig(debug_path_outpath)

    # For every plot on the plot x-axis, the indices of Q and G for the
    # nearest (Q + G)
    return {
        'Q': qg_q_ids[plot_kpoint_qg_ids],
        'G': qg_g_ids[plot_kpoint_qg_ids],
    }

def decimate_plot_x(d, decimate_x: int):
    d = dict(d)
    path_kpoint_pfracs = d.pop('path_kpoint_pfracs')
    path_x_coordinates = d.pop('path_x_coordinates')
    plot_xticks = d.pop('plot_xticks')
    plot_xticklabels = d.pop('plot_xticklabels')
    highsym_pfracs = d.pop('highsym_pfracs')
    path_q_indices = d.pop('path_q_indices')
    path_g_indices = d.pop('path_g_indices')
    assert not d, f'unhandled member: {repr(next(iter(d.keys())))}'

    highsym_indices = [i for (i, x) in enumerate(path_x_coordinates) if x in plot_xticks]

    def get_decimated_range(start, end):
        if decimate_x == 1:
            return range(start, end)

        # guarantee that a highsym index is plotted
        highsym_indices_in_range = [i for i in highsym_indices if i in range(start, end)]
        if highsym_indices_in_range:
            if len(highsym_indices_in_range) > 1:
                # we can't guarantee they'll all survive decimation
                warn('multiple highsym indices on a contiguous x region!')
            offset = (highsym_indices_in_range[0] - start) % decimate_x
            r = range(start + offset, end, decimate_x)
            assert highsym_indices_in_range[0] in r
            return r

        # otherwise, among those offsets that produce the best approximation of
        # the decimation ratio for this group...
        target_count = max(round((end - start) / decimate_x), 1)
        candidate_ranges = [range(start + i, end, decimate_x) for i in range(decimate_x)]
        candidate_ranges = [r for r in candidate_ranges if len(r) == target_count]
        assert candidate_ranges

        # ...pick the centermost.
        return candidate_ranges[len(candidate_ranges) // 2]

    get_qg = lambda i: (path_q_indices[i], path_g_indices[i])

    start = 0
    decimated_indices = []
    while start != len(path_q_indices):
        for end in range(start, len(path_q_indices)):
            if get_qg(start) != get_qg(end):
                break
        else:
            end = len(path_q_indices)

        decimated_indices.extend(get_decimated_range(start, end))
        start = end

    return {
        'path_kpoint_pfracs': path_kpoint_pfracs[decimated_indices],
        'path_x_coordinates': path_x_coordinates[decimated_indices],
        'plot_xticks': plot_xticks,
        'plot_xticklabels': plot_xticklabels,
        'highsym_pfracs': highsym_pfracs,
        'path_q_indices': path_q_indices[decimated_indices],
        'path_g_indices': path_g_indices[decimated_indices],
    }

#----------------------------------------------------------------
# Plotting

# In order to ensure that some settings can be overridden by user-supplied
# stylesheets, we implement some of the default settings as style dicts rather
# than by passing keyword arguments to the API.
MplStylesheet = tp.Union[str, tp.Dict[str, tp.Any]]

def cfg_matplotlib():
    """ Set custom matplotlib settings.  This must be done early, before we
    even do so much as construct a Normalize, or else labels will look funny.

    This function is safe to call multiple times. """
    import matplotlib
    matplotlib.rcParams.update({
        'text.latex.preamble': [r"""
\usepackage{gensymb}
\usepackage{amsmath}
\usepackage{siunitx}
"""],
        'text.usetex': True,
        'font.family': 'serif',
    })

Style = tp.Union[str, dict]
GLOBAL_CONFIG: Style = {
    'axes.labelsize': 20,
    'ytick.labelsize': 16,
    'xtick.labelsize': 16,
    'figure.figsize': (7, 8),
    'lines.markersize': 20**0.5,
}

BASELINE_CONFIG: Style = {
    'lines.markersize': 5**0.5,
    'lines.color': 'black',
}

def compute_band_plot_scatter_data(
        q_ev_frequencies: np.ndarray,
        q_ev_gpoint_probs: np.ndarray,
        path_g_indices: np.ndarray,
        path_q_indices: np.ndarray,
        path_x_coordinates: np.ndarray,
        color_info: "ColorInfo",
        coalesced_truncate: float,
        plot_coalesce_method: tp.Optional[str],
        plot_coalesce_threshold: float,
        verbose: bool = False,
):
    """
    Produces data for the final set of scatter points that will be plotted.

    Main responsibility is to handle all of the indexing mayhem:

    * Taking data indexed by Q and Ev, and reindexing it to select data points
      that lie along the plotted path.
    * Coalescing data at similar eigenvalue. (this leaves holes in the data for
      a while that require careful treatment w.r.t. indexing)
    * Filtering out points that still have miniscule probability after
      coalescing.
    """
    sizes = check_arrays(
        q_ev_frequencies = (q_ev_frequencies, ['qpoint', 'ev'], np.floating),
        q_ev_gpoint_probs = (q_ev_gpoint_probs, ['qpoint'], object),
        q_ev_gpoint_probs_row = (q_ev_gpoint_probs[0], ['ev', 'quotient'], np.floating),
        path_g_indices = (path_g_indices, ['plot-x'], np.integer),
        path_q_indices = (path_q_indices, ['plot-x'], np.integer),
        path_x_coordinates = (path_x_coordinates, ['plot-x'], np.floating),
    )

    # HACK:
    # This is used to fix the norms of probabilities.
    # A value of 0.5 assumes that all modes have 0.5 amplitude in each layer.
    # Technically can be untrue for the case where the layers completely decouple.
    maximum_probability = 0.5

    q_ev_color_data = color_info.q_ev_data()

    # At this point, it's just easier to work with ev_gpoint_probs if it is a
    # dense 2d array of sparse row vectors.
    q_g_ev_probs = np.array([
        list(ev_g_probs.T)
        for ev_g_probs in q_ev_gpoint_probs
    ])
    # this thing looks like a 2d array...
    assert q_g_ev_probs.shape == (sizes['qpoint'], sizes['quotient']), (q_g_ev_probs.shape, (sizes['qpoint'], sizes['quotient']))
    # but its elements are sparse row vectors
    assert sparse.issparse(q_g_ev_probs[0][0])
    assert q_g_ev_probs[0][0].shape[0] == 1

    # coalesce eigenmodes with too-similar frequencies
    #
    # FIXME: doing this now has the disadvantage that we may see points that are
    #        spuriously of an unusual color (where low probability modes of one
    #        color cross high probability modes of another color).
    #
    #        Ideally the point color should be a weighted mean that accounts
    #        for probability, but then we would have to do it per G point, or
    #        per plot X point.
    if plot_coalesce_method != 'none':
        if verbose:
            print('Coalescing similar frequencies...')

        q_splits = [
            list(coalesce.get_splits(ev_frequencies, plot_coalesce_threshold))
            for ev_frequencies in q_ev_frequencies
        ]

        for q_index in range(sizes['qpoint']):
            coalesce.coalesce_inplace(q_splits[q_index], q_ev_frequencies[q_index], "mean", fill=np.nan)
            if q_ev_color_data.ndim == 2:
                coalesce.coalesce_inplace(q_splits[q_index], q_ev_color_data[q_index], "max", fill=0)
            elif q_ev_color_data.ndim == 3:
                coalesce.coalesce_inplace(q_splits[q_index], q_ev_color_data[q_index], "first", fill=0)
            else:
                assert False, "illegal q_ev_color_data dim"

        q_g_ev_probs = np.array([
            [
                coalesce.coalesce_sparse_row_vec(splits, ev_probs, plot_coalesce_method)
                for ev_probs in g_ev_probs
            ] for (splits, g_ev_probs) in zip(q_splits, q_g_ev_probs)
        ])

    if verbose:
        print('Postprocessing...')

    # select the right data at each plot x point
    path_ev_probs = sparse.vstack(q_g_ev_probs[path_q_indices, path_g_indices])
    path_ev_probs = np.asarray(path_ev_probs.todense()) # it's no longer N^2 but rather X*N

    path_ev_frequencies = q_ev_frequencies[path_q_indices]
    path_ev_color_data = color_info.q_ev_data()[path_q_indices]
    assert path_ev_probs.shape == (sizes['plot-x'], sizes['ev'])
    assert path_ev_frequencies.shape == (sizes['plot-x'], sizes['ev'])

    path_ev_probs /= maximum_probability

    path_ev_mask = np.logical_and(
        path_ev_probs > coalesced_truncate, # remove probs too small to matter
        np.isfinite(path_ev_frequencies), # remove holes left behind by coalesce
    )

    # make flat arrays for the scatter plot
    X = []
    for (x_coord, ev_mask) in zip(path_x_coordinates, path_ev_mask):
        X.extend([x_coord] * ev_mask.sum())

    X = np.array(X)
    Y = path_ev_frequencies[path_ev_mask]
    Prob = path_ev_probs[path_ev_mask]
    ColorData = path_ev_color_data[path_ev_mask]

    RGB = color_info.data_to_rgb(ColorData, Prob)

    return { 'x': X, 'y': Y, 'rgb': RGB, 'prob': Prob }

def generate_band_plot(
        scatter_data: tp.Dict[str, np.ndarray],
        baseline_data: tp.Dict[str, np.ndarray],
        color_info: "ColorInfo",
        plot_xmax: float,
        plot_style: tp.List[str],
        plot_unfolded_style: tp.List[str],
        plot_baseline_style: tp.List[str],
        plot_baseline_mode: str,
        plot_baseline_color: str,
        plot_xticks: np.ndarray,
        plot_xticklabels: np.ndarray,
        plot_ylim: tp.Tuple[tp.Optional[float], tp.Optional[float]],
        plot_zone_crossing_xs: np.ndarray,
        plot_baseline_path: np.ndarray,
        plot_title: tp.Optional[str],
        plot_sidebar: bool,
        plot_colorbar: bool,
        plot_hide_unfolded: bool,
        plot_using_size: tp.Optional[str],
        plot_scatter_size: float,
        verbose: bool,
):
    import matplotlib.pyplot as plt

    X, Y = scatter_data['x'], scatter_data['y']
    RGB, Prob = scatter_data['rgb'], scatter_data['prob']
    Alpha = scatter_data['alpha']
    base_X, base_Y = baseline_data['X'], baseline_data['Y']

    if verbose:
        print(f'Plotting {len(X)} points!')

    if plot_using_size == 'only':
        Size = Alpha * plot_scatter_size
        Alpha = np.ones_like(Alpha)
    elif plot_using_size == 'both':
        Size = Prob * plot_scatter_size
    elif plot_using_size is None:
        Size = np.ones_like(Alpha) * plot_scatter_size
    else: assert False, "(BUG) invalid plot_using_size!?"

    C = np.hstack([RGB, Alpha[:, None]])

    with mpl_context([GLOBAL_CONFIG] + plot_style):
        fig = plt.figure(constrained_layout=True)
        # fig.set_tight_layout(True)

        if plot_sidebar:
            gs = fig.add_gridspec(ncols=8, nrows=1)
            ax = fig.add_subplot(gs[0, :-1])
            ax_sidebar = fig.add_subplot(gs[0, -1], sharey=ax)
        else:
            ax = fig.add_subplot(111)

        if not plot_hide_unfolded:
            with mpl_context(plot_unfolded_style):
                ax.scatter(X, Y, Size, C)
        if plot_baseline_path is not None:
            with mpl_context([BASELINE_CONFIG] + plot_baseline_style):
                base_X /= np.max(base_X)
                base_X *= np.max(X)
                if plot_baseline_mode == 'scatter':
                    ax.scatter(base_X.reshape(-1), base_Y.reshape(-1), color=plot_baseline_color)
                elif plot_baseline_mode == 'line':
                    for (base_X_row, base_Y_row) in zip(base_X, base_Y):
                        ax.plot(base_X_row, base_Y_row, color=plot_baseline_color)
                else: assert False, plot_baseline_mode

        ax.axhline(0, color='k', linestyle=':', zorder=0)

        for x in plot_xticks:
            ax.axvline(x, color='k', zorder=0)

        for x in plot_zone_crossing_xs:
            ax.axvline(x, color='k', ls=':', zorder=0)

        ax.set_xlim(0, plot_xmax)
        ax.set_xticks(plot_xticks)
        ax.set_xticklabels(plot_xticklabels)
        ax.set_ylabel('Frequency (cm$^{-1}$)')

        ax.set_ylim(*plot_ylim)

        if plot_sidebar:
            ax_sidebar.set_xlim(-1, 1)
            ax_sidebar.hlines(Y, -1, 1, color=C)
            ax_sidebar.set_xticks([0])
            ax_sidebar.set_xticklabels([r'$\mathrm{\Gamma}$'], fontsize=20)
            plt.setp(ax_sidebar.get_yticklabels(), visible=False)

        if plot_title:
            ax.set_title(plot_title)

        if plot_colorbar:
            from matplotlib import cm

            # (note: we already made sure this is not None at the beginning of the function)
            cmap, norm, cbar_label = color_info.cbar_info()

            # Because we modify alpha independently of the color, there's no way
            # for scatter to use a colormap (so we didn't even try). Rather, we made
            # a norm object, which we can use in an empty mappable to give colorbar
            # something to work with.
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(np.array([]))
            cbar = plt.colorbar(sm, ax=ax, aspect=50)
            cbar.ax.tick_params(labelsize='x-large')
            cbar.set_label(cbar_label, size='xx-large')

    # end context manager

    return fig, ax

class ColorInfo:
    def cbar_info(self):
        """ Get ``(cmap, norm, cbar_label)`` for a colorbar, or ``None``. """
        return None

    def q_ev_data(self):
        """
        Get data for colorizing which will be at qpoint and eigenvector.

        The output has shape ``(nquotient, nev, ...)``, where the remaining
        dimensions may vary between different instances of ``ColorInfo``.
        """
        raise NotImplementedError

    def data_to_rgb(self, data, probs):
        """
        Get the rgb values for a color array.

        :param data: Shape ``(ndata, ...)`` array of subarrays taken from
        ``self.q_ev_data()`` at visible data points.
        :param probs: Shape ``(ndata,)`` array of projection probabilities.
        Used by one of the coloring modes.
        :return: Shape ``(ndata, 3)`` array. May point into ``Data``.
        """
        raise NotImplementedError

class RgbColorInfo(ColorInfo):
    def __init__(self, q_ev_rgb):
        self.__q_ev_rgb = q_ev_rgb

    def q_ev_data(self): return self.__q_ev_rgb
    def data_to_rgb(self, data, probs): return data

class CmapColorInfo(ColorInfo):
    def __init__(self, data, cmap, norm, cbar_label):
        self.__q_ev_data = data
        self.__cmap = cmap
        self.__norm = norm
        self.__cbar_label = cbar_label

    def cbar_info(self): return self.__cmap, self.__norm, self.__cbar_label
    def q_ev_data(self): return self.__q_ev_data
    def data_to_rgb(self, data, probs): return self.__cmap(self.__norm(data))[:, :3]

# This one has to have its own implementation because it cannot possibly store
# data indexed by Q and eV due to the G points.
# (it has to wait until we are indexing by plot x)
class ProbCmapColorInfo(ColorInfo):
    def __init__(self, shape, cmap, norm, cbar_label):
        self.__dummy_data = np.zeros(shape)
        self.__norm = norm
        self.__cmap = cmap
        self.__cbar_label = cbar_label

    def cbar_info(self): return self.__cmap, self.__norm, self.__cbar_label
    def q_ev_data(self): return self.__dummy_data
    def data_to_rgb(self, data, probs): return self.__cmap(self.__norm(probs))[:, :3]

def get_plot_color_info(plot_color_string, z_pol, raman_dict) -> ColorInfo:
    from matplotlib import colors, cm

    if ':' in plot_color_string:
        mode, _arg = plot_color_string.split(':', 2)
    else:
        mode, _arg = plot_color_string, None

    def expect_no_arg():
        if _arg is not None:
            die(f'--plot-color={mode} takes no argument')

    def expect_arg():
        if _arg is None:
            die(f'--plot-color={mode} requires an argument')
        return _arg

    if mode == 'zpol':
        expect_no_arg()
        cmap = colors.LinearSegmentedColormap.from_list('', [[0, 0, 1], [0, 0.5, 0]])
        norm = colors.Normalize(vmin=0, vmax=1)
        label = 'z-polarization'
        return CmapColorInfo(z_pol, cmap, norm, label)

    elif mode == 'uniform':
        fixed_color = colors.to_rgb(expect_arg().strip())
        color = np.zeros(z_pol.shape + (3,))
        color[...] = fixed_color
        return RgbColorInfo(color)

    elif mode == 'raman':
        raman_dict_key = expect_arg()
        if raman_dict_key not in ['average-3d', 'backscatter']:
            die('Invalid raman polarization mode: {raman_dict_key}')

        data = raman_dict[raman_dict_key]
        data /= data.max()
        data = np.log10(np.maximum(data, 1e-7))
        label = r'$\mathrm{log}_{10}\left(\text{Raman intensity / max}\right)$'

        cmap = cm.get_cmap('cool_r')
        norm = colors.Normalize(vmin=data.min(), vmax=data.max())
        return CmapColorInfo(data, cmap, norm, label)

    elif mode == 'prob':
        cmap_name = expect_arg()
        label = r'Unfolding probability'

        cmap = cm.get_cmap(cmap_name)
        norm = colors.Normalize(vmin=0., vmax=1.)
        return ProbCmapColorInfo(z_pol.shape, cmap, norm, label)

    else:
        die(f'invalid --plot-color mode: {repr(mode)}')
        raise RuntimeError('unreachable')

def get_parallelogram_zone_crossings(
        supercell,
        highsym_pfracs,
        plot_xticks,
):
    check_arrays(
        highsym_pfracs = (highsym_pfracs, ['special_point', 3], np.floating),
        plot_xticks = (plot_xticks, ['special_point'], np.floating),
    )

    highsym_sfracs = highsym_pfracs @ supercell.matrix.T

    out = []
    for ((pred_x, pred_sfrac), (succ_x, succ_sfrac)) in window2(zip(plot_xticks, highsym_sfracs)):
        # Skip discontiguous regions in kpath (denoted by the same x coord appearing twice in a row)
        if pred_x == succ_x:
            continue

        # consider one family of plane boundaries at a time
        for k in range(3):
            # find integer values (plane indices) of the fractional coordinate on this line
            min_coord, max_coord = sorted([pred_sfrac[k], succ_sfrac[k]])
            crossed_ints = np.arange(np.floor(min_coord), np.ceil(max_coord))
            if not crossed_ints.size:
                continue # no boundaries crossed; avoid potential division by zero

            # Solve the linear equation between this fractional coordinate and the plot x axis
            # to find the x values of those planes
            m = (succ_x - pred_x) / (succ_sfrac[k] - pred_sfrac[k])
            out.extend(pred_x + m * (crossed_ints - pred_sfrac[k]))

    return np.array(sorted(out))

def read_baseline_plot(path):
    X, Y = [], []
    d = dwim.from_path(path)
    d = d['phonon']
    for qpoint in d:
        X.append([qpoint['distance']] * len(qpoint['band']))
        Y.append([band['frequency'] * THZ_TO_WAVENUMBER for band in qpoint['band']])
    # order by band, then frequency
    X = np.array(X).T
    Y = np.array(Y).T
    return X, Y

#---------------------------------------------------------------
# debugging

def debug_bin_magnitudes(array):
    from collections import Counter

    zero_count = product(array.shape) - np.sum(array != 0)
    if sparse.issparse(array):
        array = array.data
    array = array[array != 0]
    logs = np.floor(np.log10(array)).astype(int)
    counter = Counter(logs)
    counter[-999] = zero_count
    print("Magnitude summary:")
    print(sorted(counter.items()))

def debug_quotient_points(points2, lattice2):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 8))
    draw_unit_cell(ax, lattice2, lw=2)
    draw_reduced_points(ax, points2, lattice2)
    ax.set_aspect('equal', 'box')
    plt.show()

def debug_path(points, lattice, path, supercell=None):
    import matplotlib.pyplot as plt

    points = unfold_lib.reduce_carts(points, lattice)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(points[:, 0], points[:, 1])
    ax.set(aspect='equal', adjustable='box')
    if supercell:
        prim_lattice = np.linalg.inv(supercell.matrix)[:2, :2] @ lattice[:2, :2]
        draw_prim_cell_boundaries(ax, prim_lattice, lattice, ls=':', color='k')
        draw_axis_vectors(ax, prim_lattice)

    draw_unit_cell(ax, lattice[:2, :2], lw=2)
    draw_path(ax, path, ls=':', lw=2)

    cell_path = lattice_path(lattice[:2, :2])
    ax.set_xlim(add_fuzz_to_interval((min(cell_path[:, 0]), max(cell_path[:, 0])), 0.05))
    ax.set_ylim(add_fuzz_to_interval((min(cell_path[:, 1]), max(cell_path[:, 1])), 0.05))

    return fig, ax

def draw_axis_vectors(ax, lattice, **kw):
    from matplotlib.collections import LineCollection

    lines = LineCollection(
        np.array(np.array([[(0, 0), (1, 0)], [(0, 0), (0, 1)]]) @ lattice),
        colors=[(1, 0, 0, 1), (0, 0, 1, 1)],
        **kw
    )
    ax.add_collection(lines)
    return lines

def draw_prim_cell_boundaries(ax, prim_lattice, super_lattice, **kw):
    import itertools
    # NOTE: shapely is not listed in requirements.txt because it is annoying
    #       to install, and this function is only ever used for debugging.
    from shapely.geometry import Polygon, LineString, Point
    from matplotlib.collections import LineCollection

    super_lattice = super_lattice[:2, :2]

    # Inverses of matrices of the form  [ A  -a ].T, where a is a primitive
    # lattice basis vector and A is a superlattice basis vector.
    solvers = [[None for _ in range(2)] for _ in range(2)]
    for time_axis in range(2):
        for super_time_axis in range(2):
            matrix = np.array([
                super_lattice[super_time_axis],
                -prim_lattice[time_axis],
            ])

            if abs(np.linalg.det(matrix)) < 1e-4:
                continue # leave None in the list

            solvers[time_axis][super_time_axis] = np.linalg.inv(matrix)

    super_polygon = Polygon(np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]) @ super_lattice)

    # Draw lines of the form  'n a + t b', or 'n b + t a'
    # (where n is an integer, and a and b are the basis vectors),
    # considering either only non-negative n or only negative n.
    segments = []
    for int_axis, time_axis in [(0, 1), (1, 0)]:
        for n_start, n_step in [(0, 1), (-1, -1)]:
            for n in itertools.count(n_start, n_step):
                overly_long_line_segment = [
                    n * prim_lattice[int_axis] + t * prim_lattice[time_axis]
                    for t in [-999999, 999999]
                ]

                # Use shapely to find the true intersections
                true_intersection = super_polygon.intersection(LineString(overly_long_line_segment))

                # If there were no intersections, we are outside the cell.
                # We don't need to consider values of n further in this direction.
                if true_intersection.is_empty:
                    break

                if isinstance(true_intersection, Point):
                    continue # "tangent" to polygon? could easily happen for n = 0

                assert isinstance(true_intersection, LineString)
                segments.append(true_intersection.coords)

    lines = LineCollection(segments, **kw)
    ax.add_collection(lines)
    return lines

def draw_path(ax, path, **kw):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    mpl_path = Path(vertices=np.array(path))
    return ax.add_patch(patches.PathPatch(mpl_path, facecolor='none', **kw))

def draw_unit_cell(ax, lattice2, **kw):
    return draw_path(ax, lattice_path(lattice2), **kw)

def lattice_path(lattice2):
    np.testing.assert_equal(lattice2.shape, [2,2])
    return np.array([[0,0], lattice2[0], lattice2[0]+lattice2[1], lattice2[1], [0,0]])

def draw_reduced_points(ax, points2, lattice2, **kw):
    points2 = unfold_lib.reduce_carts(points2, lattice2)
    ax.scatter(points2[:, 0], points2[:, 1], **kw)

#---------------------------------------------------------------
# CLI behavior

def parse_qpoint(s):
    def parse_number(word):
        try:
            if '/' in word:
                numer, denom = (int(x.strip()) for x in word.split('/'))
                return numer / denom
            else:
                return float(word.strip())
        except ValueError:
            raise ValueError(f'{repr(word)} is not an integer, float, or rational number')

    if '[' in s:
        warn('JSON input for --qpoint is deprecated; use a whitespace separated list of numbers.')
        lst = [1.0 * x for x in json.loads(s)]
    else:
        lst = [parse_number(word) for word in s.split()]

    if len(lst) != 3:
        raise ValueError('--qpoint must be of dimension 3')

    return lst

def parse_ylim(s):
    def maybe_float(word):
        word = word.strip()
        if word: return float(word)
        else: return None
    tup = tuple(map(maybe_float, s.split(':')))
    if len(tup) != 2:
        raise ValueError('ylim must be of form MIN:MAX')
    return tup

def parse_figsize(s):
    tup = tuple(float(x.strip()) for x in s.split('x'))
    if len(tup) != 2:
        raise ValueError('size must be of form WxH')
    return tup

def check_optional_input(path):
    if path is not None and not os.path.exists(path):
        die(f'Does not exist: \'{path}\'')

def check_optional_output_ext(argument, path, only=None, forbid=None):
    """ Validate the extension for an output file.

    Because this script uses DWIM facilities, some arguments support many possible
    filetypes.  However, there are some cases where it's easy to forget whether
    something should be .npy or .npz.  Calling this function with `forbid=` in this
    case can be helpful.
    """
    if path is None:
        return

    if only is None and forbid is None:
        raise TypeError('must supply only or forbid')

    if forbid is not None:
        if isinstance(forbid, str):
            forbid = [forbid]

        for ext in forbid:
            if path.endswith(ext) or path.endswith(ext + '.gz') or path.endswith(ext + '.xz'):
                die(f'Invalid extension for {argument}: {path}')

    if only is not None:
        if isinstance(only, str):
            only = [only]

        if not any(path.endswith(ext) for ext in only):
            expected = ', '.join(only)
            die(f'Invalid extension for {argument}: expected one of: {expected}')

#---------------------------------------------------------------
# utils

def mpl_context(arg: tp.Union[Style, tp.List[Style]]):
    """ ``matplotlib.pyplot.style.context`` with a better type annotation. """
    import matplotlib.pyplot as plt
    return plt.style.context(arg)

def window2(xs):
    prev = next(xs)
    for x in xs:
        yield (prev, x)
        prev = x

def product(iter):
    from functools import reduce
    return reduce((lambda a, b: a * b), iter)

def truncate(array, tol):
    array = array.copy()
    if sparse.issparse(array):
        data = array.data
        data[np.absolute(data) < tol] = 0.0
        return array
    else:
        array[np.absolute(array) < tol] = 0.0
        return array

def add_fuzz_to_interval(ivl, fuzz):
    a, b = ivl
    return a + (a - b) * fuzz, b + (b - a) * fuzz

def dict_zip(*dicts):
    """
    Take a series of dicts that share the same keys, and reduce the values
    for each key as if folding an iterator.
    """
    keyset = set(dicts[0])
    for d in dicts:
        if set(d) != keyset:
            raise KeyError(f"Mismatched keysets in fold_dicts: {sorted(keyset)}, {sorted(set(d))}")

    return { key: [d[key] for d in dicts] for key in keyset }

def to_slice_if_possible(mask):
    """
    Take a 1D boolean array intended for use in advanced indexing, and either
    return the input mask unchanged, or a ``slice`` with equivalent meaning.

    This can allow reduced memory usage, as a ``slice`` enables basic indexing
    (avoiding copies).
    """
    assert mask.ndim == 1
    indices, = np.where(mask)
    unique_diffs = set(indices[1:] - indices[:-1])
    if len(unique_diffs) == 1:
        return slice(indices[0], indices[-1] + 1, unique_diffs.pop())
    else:
        return mask

#---------------------------------------------------------------

def warn(*args, **kw):
    print('unfold:', *args, **kw, file=sys.stderr)

def die(*args, **kw):
    warn(*args, **kw)
    if SHOW_ACTION_STACK:
        for name in ACTION_STACK[::-1]:
            print(f"  while computing {name}", file=sys.stderr)
    sys.exit(1)

#---------------------------------------------------------------

if __name__ == '__main__':
    main()
