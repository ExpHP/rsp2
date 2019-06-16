#!/usr/bin/env python3

import os
import numpy as np
import json
import sys
import typing as tp
from scipy import interpolate as scint
from scipy import sparse
from pymatgen import Structure

THZ_TO_WAVENUMBER = 33.3564095198152

try:
    import rsp2
except ImportError:
    info = lambda s: print(s, file=sys.stderr)
    info('Please add the following to your PYTHONPATH:')
    info('  (rsp2 source root)/scripts')
    info('  (rsp2 source root)/src/python')
    sys.exit(1)

from rsp2.io import eigensols, structure_dir, dwim

DEFAULT_TOL = 1e-2

A = tp.TypeVar('A')
B = tp.TypeVar('B')

def main():
    global SHOW_ACTION_STACK

    import argparse

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

    structure = register(TaskStructure())

    kpoint_sfrac = register(TaskKpointSfrac())

    dynmat = register(TaskDynmat())

    eigensols = register(TaskEigensols(structure, dynmat))

    translation_deperms = register(TaskDeperms(structure))

    ev_gpoint_probs = register(TaskQProbs(structure, kpoint_sfrac, eigensols, translation_deperms))

    band_path = register(TaskBandPath(structure))

    bands = register(TaskBands(structure, kpoint_sfrac, band_path, ev_gpoint_probs))

    mode_data = register(TaskEigenmodeData(eigensols))

    _bandplot = register(TaskBandPlot(structure, mode_data, ev_gpoint_probs, band_path, bands))

    for task in all_tasks:
        task.add_parser_opts(parser)

    args = parser.parse_args()

    if not any(task.has_action(args) for task in all_tasks):
        parser.error("Nothing to do!")

    if args.debug:
        SHOW_ACTION_STACK = True

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

class Task:
    NOT_YET_COMPUTED = object()

    def __init__(self):
        self.cached = Task.NOT_YET_COMPUTED

    def add_parser_opts(self, parser):
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

class TaskKpointSfrac(Task):
    def add_parser_opts(self, parser):
        parser.add_argument(
            '--kpoint', default='[0,0,0]', help=
            'kpoint in fractional coordinates of the superstructure reciprocal '
            'cell, as a json array of 3 floats',
        )

    def _compute(self, args):
        return [1.0 * x for x in json.loads(args.kpoint)]

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

    def _compute(self, args, **_kw):
        layer = args.layer

        if not os.path.isdir(args.STRUCTURE):
            die('currently, only rsp2 structure directory format is supported')

        sdir = structure_dir.from_path(args.STRUCTURE)
        structure = sdir.structure
        if sdir.layer_sc_matrices is None:
            die("the structure must supply layer-sc-matrices")
        supercell = Supercell(sdir.layer_sc_matrices[args.layer])

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

    def has_action(self, args):
        return bool(args.write_perms)

    def _compute(self, args):
        if args.perms:
            return np.array(dwim.from_path(args.perms))
        else:
            progress_callback = None
            if args.verbose:
                def progress_callback(done, count):
                    print(f'Deperms: {done:>5} of {count} translations')

            return collect_translation_deperms(
                superstructure=self.structure.require(args)['projected_structure'],
                supercell=self.structure.require(args)['supercell'],
                axis_mask=np.array([1,1,0]),
                tol=DEFAULT_TOL,
                progress=progress_callback,
            )

    def _do_action(self, args):
        if args.write_perms:
            translation_deperms = self.require(args)
            dwim.to_path(args.write_perms, translation_deperms)

class TaskDynmat(Task):
    def add_parser_opts(self, parser):
        parser.add_argument(
            '--dynmat', metavar='FILE', help='rsp2 dynmat file (.npz)',
        )

    def _compute(self, args):
        if not args.dynmat:
            die('--dynmat is required for this action')
        return sparse.load_npz(args.dynmat)

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
            import scipy.linalg
            if args.verbose:
                print('--eigensols not supplied. Will diagonalize dynmat.')

            m = self.dynmat.require(args)
            ev_eigenvalues, ev_eigenvectors = scipy.linalg.eigh(m.todense())
            ev_eigenvectors = ev_eigenvectors.T

        ev_projected_eigenvectors = ev_eigenvectors.reshape((-1, nsites, 3))[:, mask]

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

    def has_action(self, args):
        return bool(args.write_mode_data)

    def _compute(self, args):
        if args.mode_data:
            npz = np.load(args.mode_data)
            return {
                'ev_frequencies': npz.f.ev_frequencies,
                'ev_z_projections': npz.f.ev_z_projections,
            }
        else:
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

# Arguments related to probabilities
# (an intermediate file format that can be significantly smaller than the
#  input eigenvectors and thus easier to transmit)
class TaskQProbs(Task):
    def __init__(
            self,
            structure: TaskStructure,
            kpoint_sfrac: TaskKpointSfrac,
            eigensols: TaskEigensols,
            translation_deperms: TaskDeperms,
    ):
        super().__init__()
        self.structure = structure
        self.kpoint_sfrac = kpoint_sfrac
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
            '--probs', metavar='FILE', help=
            'Path to .npz file previously written through --write-probs.',
        )

    def has_action(self, args):
        return bool(args.write_probs)

    def _compute(self, args):
        if args.probs:
            ev_gpoint_probs = dwim.from_path(args.probs)
        else:
            if args.verbose:
                print('--probs not supplied. Will compute by unfolding eigensols.')

            layer = self.structure.require(args)['layer']

            progress_callback = None
            if args.verbose:
                def progress_callback(done, count):
                    print(f'Layer {layer}: Unfolding {done:>5} of {count} eigenvectors')

            # reading the file might take forever; compute deperms first as it has
            # a greater chance of having a bug
            self.translation_deperms.require(args)

            ev_gpoint_probs = unfold_all(
                superstructure=self.structure.require(args)['projected_structure'],
                supercell=self.structure.require(args)['supercell'],
                eigenvectors=self.eigensols.require(args)['ev_projected_eigenvectors'],
                kpoint_sfrac=self.kpoint_sfrac.require(args),
                translation_deperms=self.translation_deperms.require(args),
                progress=progress_callback,
            )

        if args.verbose:
            debug_bin_magnitudes(ev_gpoint_probs)

        ev_gpoint_probs = truncate(ev_gpoint_probs, args.probs_threshold)
        ev_gpoint_probs = sparse.csr_matrix(ev_gpoint_probs)

        if args.verbose:
            density = ev_gpoint_probs.nnz / product(ev_gpoint_probs.shape)
            print('Probs matrix density: {:.4g}%'.format(100.0 * density))

        return ev_gpoint_probs

    def _do_action(self, args):
        ev_gpoint_probs = self.require(args)
        if args.write_probs:
            dwim.to_path(args.write_probs, ev_gpoint_probs)

class TaskBandPath(Task):
    def __init__(
            self,
            structure: TaskStructure,
    ):
        super().__init__()
        self.structure = structure

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--plot-kpath', help=
            "A kpath in the format accepted by ASE's parse_path_string, "
            "naming points in the monolayer BZ.  If not specified, no band "
            "plot is generated."
        )

    def _compute(self, args):
        from ase.dft.kpoints import bandpath, parse_path_string

        supercell = self.structure.require(args)['supercell']
        super_lattice = self.structure.require(args)['structure'].lattice.matrix

        prim_lattice = np.linalg.inv(supercell.matrix) @ super_lattice

        if args.plot_kpath is None:
            die('--plot-kpath is required')

        # NOTE: The kpoints returned by get_special_points (and by proxy, this
        #       function) do adapt to the user's specific choice of primitive cell.
        #       (at least, for reasonable cells; I haven't tested it with a highly
        #       skewed cell). Respect!
        bandpath_output = bandpath(args.plot_kpath, prim_lattice, 300)

        point_names = parse_path_string(args.plot_kpath)
        if len(point_names) > 1:
            die('This script currently does not support plots along discontinuous paths.')
        point_names, = point_names

        point_names = [r'$\mathrm{\Gamma}$' if x == 'G' else x for x in point_names]

        return {
            'plot_kpoint_pfracs': bandpath_output[0],
            'plot_x_coordinates': bandpath_output[1],
            'plot_xticks': bandpath_output[2],
            'plot_xticklabels': point_names,
        }

# Arguments related to high symmetry path resampling
# (an intermediate file format that allows skipping the griddata computations)
class TaskBands(Task):
    def __init__(
            self,
            structure: TaskStructure,
            kpoint_sfrac: TaskKpointSfrac,
            band_path: TaskBandPath,
            ev_gpoint_probs: TaskQProbs,
    ):
        super().__init__()
        self.structure = structure
        self.band_path = band_path
        self.ev_gpoint_probs = ev_gpoint_probs
        self.kpoint_sfrac = kpoint_sfrac

    def add_parser_opts(self, parser):
        parser.add_argument(
            '--write-bands', metavar='FILE', help=
            'Write data resampled onto layer high sym path. (.npz)',
        )

        parser.add_argument(
            '--bands', metavar='FILE', help=
            'Path to file previously written through --write-bands.',
        )

    def has_action(self, args):
        return bool(args.write_bands)

    def _compute(self, args):
        if args.bands:
            ev_path_probs = dwim.from_path(args.bands)
        else:
            progress_callback = None
            if args.verbose:
                print('--bands not supplied. Will compute by resampling from Q probabilities.')
                def progress_callback(done, count):
                    print(f'Resampling: {done:>5} of {count} eigenvectors')

            ev_path_probs = resample_gprobs_on_kpath(
                    super_lattice=self.structure.require(args)['structure'].lattice.matrix,
                    supercell=self.structure.require(args)['supercell'],
                    ev_gpoint_probs=self.ev_gpoint_probs.require(args),
                    kpoint_sfrac=self.kpoint_sfrac.require(args),
                    plot_kpoint_pfracs=self.band_path.require(args)['plot_kpoint_pfracs'],
                    progress=progress_callback,
            )

        return { 'ev_path_probs': ev_path_probs }

    def _do_action(self, args):
        if args.write_bands:
            ev_path_probs = self.require(args)['ev_path_probs']
            dwim.to_path(args.write_bands, ev_path_probs)

class TaskBandPlot(Task):
    def __init__(
            self,
            structure: TaskStructure,
            mode_data: TaskEigenmodeData,
            ev_gpoint_probs: TaskQProbs,
            band_path: TaskBandPath,
            bands: TaskBands,
    ):
        super().__init__()
        self.structure = structure
        self.mode_data = mode_data
        self.ev_gpoint_probs = ev_gpoint_probs
        self.band_path = band_path
        self.bands = bands

    def add_parser_opts(self, parser):
        parser.add_argument('--show', action='store_true', help='show plot')
        parser.add_argument('--write-plot', metavar='FILE', help='save plot to file')

        parser.add_argument(
            '--plot-exponent', type=float, metavar='VALUE', default=1.0, help=
            'Scale probabilities by this exponent before plotting.'
        )

        parser.add_argument(
            '--plot-max-alpha', type=float, metavar='VALUE', default=1.0, help=
            'Scale probabilities by this exponent before plotting.'
        )

        parser.add_argument(
            '--plot-truncate', type=float, metavar='VALUE', default=0.0, help=
            'Don\'t plot points whose final alpha is less than this. '
            'This can be a good idea for SVG and PDF outputs.'
        )

        parser.add_argument(
            '--plot-baseline-file', type=str, metavar='FILE', help=
            'Data file for a "normal" plot.  Phonopy band.yaml is accepted.'
        )

        parser.add_argument(
            '--plot-color', type=str, default='zpol', metavar='FILE', help=
            'How the plot points are colored. Choices: [zpol, sign, uniform:COLOR] '
            '(e.g. --plot-color uniform:blue)'
        )

        parser.add_argument(
            '--plot-sidebar', action='store_true', help=
            'Show a sidebar with the frequencies all on the same point.'
        )

        parser.add_argument(
            '--plot-ylim', help=
            'Set plot ylim.'
        )

        parser.add_argument(
            '--plot-hide-unfolded', action='store_true', help=
            "Don't actually show the unfolded probs. (intended for use with --plot-baseline-file, "
            "so that you can show only the baseline)"
        )

    def has_action(self, args):
        return args.show or bool(args.write_plot)

    def _compute(self, args):
        return probs_to_band_plot(
            ev_frequencies=self.mode_data.require(args)['ev_frequencies'],
            ev_z_projections=self.mode_data.require(args)['ev_z_projections'],
            ev_path_probs=self.bands.require(args)['ev_path_probs'],
            plot_x_coordinates=self.band_path.require(args)['plot_x_coordinates'],
            plot_xticks=self.band_path.require(args)['plot_xticks'],
            plot_xticklabels=self.band_path.require(args)['plot_xticklabels'],
            alpha_exponent=args.plot_exponent,
            alpha_max=args.plot_max_alpha,
            alpha_truncate=args.plot_truncate,
            plot_baseline_path=args.plot_baseline_file,
            plot_color=args.plot_color,
            plot_ylim=args.plot_ylim,
            plot_sidebar=args.plot_sidebar,
            plot_hide_unfolded=args.plot_hide_unfolded,
            verbose=args.verbose,
        )

    def _do_action(self, args):
        fig, ax = self.require(args)

        if args.show:
            import matplotlib.pyplot as plt
            # fig.show() # doesn't do anything :/
            plt.show()

        if args.write_plot:
            fig.savefig(args.write_plot)

#----------------------------------------------------------------

def resample_gprobs_on_kpath(
        super_lattice,
        supercell,
        ev_gpoint_probs,
        kpoint_sfrac,
        plot_kpoint_pfracs,
        progress=None,
):
    check_arrays(
        super_lattice = (super_lattice, [3, 3], np.floating),
        ev_gpoint_probs = (ev_gpoint_probs, ['ev', 'quotient'], np.floating),
        kpoint_sfrac = (kpoint_sfrac, [3], np.floating),
        plot_kpoint_pfracs = (plot_kpoint_pfracs, ['x', 3], np.floating),
    )

    prim_lattice = np.linalg.inv(supercell.matrix) @ super_lattice

    # Grid of kpoints at which probabilities were computed
    kpoint_sfracs = supercell.gpoint_sfracs() + kpoint_sfrac
    kpoint_carts = kpoint_sfracs @ np.linalg.inv(super_lattice).T

    plot_kpoint_carts = plot_kpoint_pfracs @ np.linalg.inv(prim_lattice).T

    ev_path_probs = []
    for gpoint_probs in iter_with_progress(ev_gpoint_probs, progress):
        if sparse.issparse(gpoint_probs):
            gpoint_probs = np.asarray(gpoint_probs.todense()).squeeze(axis=0)

        # Here we encounter a big problem:
        #
        #     The points at which we want to draw bands are not necessarily
        #     images of the kpoint at which we computed eigenvectors.
        #
        # Our solution is not very rigorous.
        #
        # For each band, we will take its probabilities at each discrete image
        # of the kpoint, and use them to define a surface, performing
        # interpolation in the areas in-between.
        #
        # This is obviously not correct, but the idea is that for large
        # supercells, every point in the primitive BZ is close to an image of
        # the supercell gamma point, limiting the negative impact of this
        # interpolation scheme... I hope.
        ev_path_probs.append(griddata_periodic(
            points=kpoint_carts,
            values=gpoint_probs,
            xi=plot_kpoint_carts,
            lattice=np.linalg.inv(prim_lattice).T,
            periodic_axis_mask=[1,1,0], # FIXME
            method='nearest',
        ))

    return sparse.csr_matrix(ev_path_probs)

def probs_to_band_plot(
        ev_frequencies,
        ev_z_projections,
        ev_path_probs,
        plot_x_coordinates,
        plot_xticks,
        plot_xticklabels,
        plot_color,
        plot_ylim,
        alpha_truncate,
        alpha_exponent,
        alpha_max,
        plot_baseline_path,
        plot_hide_unfolded,
        plot_sidebar,
        verbose=False,
):
    import matplotlib.pyplot as plt

    if plot_baseline_path is not None:
        base_X, base_Y = read_baseline_plot(plot_baseline_path)
    else:
        base_X, base_Y = [], []

    check_arrays(
        ev_frequencies = (ev_frequencies, ['ev'], np.floating),
        ev_z_projections = (ev_z_projections, ['ev'], np.floating),
        ev_path_probs = (ev_path_probs, ['ev', 'x'], np.floating),
        plot_x_coordinates = (plot_x_coordinates, ['x'], np.floating),
        plot_xticks = (plot_xticks, ['special_point'], np.floating),
    )

    X, Y, S, Z_proj = [], [] ,[], []
    iterator = zip(ev_frequencies, ev_z_projections, ev_path_probs)
    for (frequency, z_projection, path_probs) in iterator:
        if sparse.issparse(path_probs):
            path_probs = np.asarray(path_probs.todense()).squeeze(axis=0)

        # Don't ask matplotlib to draw thousands of points with alpha=0
        mask = path_probs != 0

        X.append(plot_x_coordinates[mask])
        Y.append([frequency] * mask.sum())
        S.append(path_probs[mask])
        Z_proj.append([z_projection] * mask.sum())

    X = np.hstack(X)
    Y = np.hstack(Y)
    S = np.hstack(S)
    Z_proj = np.hstack(Z_proj)

    S **= alpha_exponent
    S *= alpha_max

    mask = S > alpha_truncate
    X = X[mask]
    Y = Y[mask]
    S = S[mask]
    Z_proj = Z_proj[mask]

    if verbose:
        print(f'Plotting {len(X)} points!')

    C = np.hstack([np.zeros((len(S), 3)), S[:, None]])

    if plot_color == 'zpol':
        # colorize Z projection
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('', [[0, 0, 1], [0, 0.5, 0]])
        C[:, :3] = cmap(Z_proj)[:, :3]
    elif plot_color == 'sign':
        from matplotlib.colors import to_rgb
        C[:,:3] = to_rgb('y')
        C[:,:3] = np.where((Y < -1e-3)[:, None], to_rgb('g'), C[:,:3])
        C[:,:3] = np.where((Y > +1e-3)[:, None], to_rgb('r'), C[:,:3])
    elif plot_color.startswith('uniform:'):
        from matplotlib.colors import to_rgb
        # use given color
        C[:, :3] = to_rgb(plot_color[len('uniform:'):].strip())
    else:
        die(f'invalid --plot-color: {repr(plot_color)}'),

    fig = plt.figure(figsize=(7, 8), constrained_layout=True)
    #fig.set_tight_layout(True)

    if plot_sidebar:
        gs = fig.add_gridspec(ncols=8, nrows=1)
        ax = fig.add_subplot(gs[0,:-1])
        ax_sidebar = fig.add_subplot(gs[0,-1], sharey=ax)
    else:
        ax = fig.add_subplot(111)

    if not plot_hide_unfolded:
        ax.scatter(X, Y, 20, C)
    if plot_baseline_path is not None:
        base_X /= np.max(base_X)
        base_X *= np.max(X)
        ax.scatter(base_X, base_Y, 5, 'k')

    for x in plot_xticks:
        ax.axvline(x, color='k')

    ax.set_xlim(X.min(), X.max())
    ax.set_xticks(plot_xticks)
    ax.set_xticklabels(plot_xticklabels, fontsize=20)
    ax.set_ylabel('Frequency (cm$^{-1}$)', fontsize=20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    if plot_ylim is not None:
        ymin, ymax = (float(x.strip()) for x in plot_ylim.split(':'))
        ax.set_ylim(ymin, ymax)

    if plot_sidebar:
        ax_sidebar.set_xlim(-1, 1)
        ax_sidebar.hlines(Y, -1, 1, color=C)
        ax_sidebar.set_xticks([0])
        ax_sidebar.set_xticklabels([r'$\mathrm{\Gamma}$'], fontsize=20)
        plt.setp(ax_sidebar.get_yticklabels(), visible=False)

    return fig, ax

def read_baseline_plot(path):
    X, Y = [], []
    d = dwim.from_path(path)
    d = d['phonon']
    for qpoint in d:
        X.extend([qpoint['distance']] * len(qpoint['band']))
        Y.extend(band['frequency'] * THZ_TO_WAVENUMBER for band in qpoint['band'])
    return X, Y

def reduce_carts(carts, lattice):
    fracs = carts @ np.linalg.inv(lattice)
    fracs %= 1.0
    fracs %= 1.0 # for values like -1e-20
    return fracs @ lattice

class Supercell:
    def __init__(self, matrix):
        """
        :param matrix: Shape ``(3, 3)``, integer.
        Integer matrix satisfying
        ``matrix @ prim_lattice_matrix == super_lattice_matrix``
        where the lattice matrices are understood to store a lattice primitive
        translation in each row.
        """
        if isinstance(matrix, Supercell):
            self.matrix = matrix.matrix
            self.repeats = matrix.repeats
            self.t_repeats = matrix.t_repeats
        else:
            matrix = np.array(matrix, dtype=int)
            assert matrix.shape == (3, 3)
            self.matrix = matrix
            self.repeats = find_repeats(matrix)
            self.t_repeats = find_repeats(matrix.T)

    def translation_pfracs(self):
        """
        :return: Shape ``(quotient, 3)``, integral.

        Fractional coordinates of quotient-space translations,
        in units of the primitive cell lattice basis vectors.
        """
        return cartesian_product(*(np.arange(n) for n in self.repeats)).astype(float)

    def gpoint_sfracs(self):
        """
        :return: Shape ``(quotient, 3)``, integral.

        Fractional coordinates of quotient-space gpoints,
        in units of the supercell reciprocal lattice basis vectors.
        """
        return cartesian_product(*(np.arange(n) for n in self.t_repeats)).astype(float)

def collect_translation_deperms(
        superstructure: Structure,
        supercell: Supercell,
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

def unfold_all(
        superstructure: Structure,
        supercell: Supercell,
        eigenvectors,
        kpoint_sfrac,
        translation_deperms,
        progress = None,
):
    """
    :param superstructure: ``pymatgen.Structure`` object with `sites` sites.
    :param supercell: ``Supercell`` object.
    :param eigenvectors: Shape ``(num_evecs, 3 * sites)``, complex.

    Each row is an eigenvector.  Their norms may be less than 1, if the
    structure has been projected onto a single layer, but should not exceed 1.
    (They will NOT be automatically normalized by this function, as projection
    onto a layer may create eigenvectors of zero norm)

    :param translation_deperms:  Shape ``(quotient, sites)``.
    Permutations such that ``(carts + translation_carts[i])[deperms[i]]`` is
    equivalent to ``carts`` under superlattice translational symmetry, where
    ``carts`` is the supercell carts.

    :param kpoint_sfrac: Shape ``(3,)``, real.
    The K point in the SC reciprocal cell at which the eigenvector was computed,
    in fractional coords.

    :param progress: Progress callback.
    Called as ``progress(num_done, num_total)``.

    :return: Shape ``(num_evecs, quotient)``
    For each vector in ``eigenvectors``, its projected probabilities
    onto ``k + g`` for each g in ``supercell.gpoint_sfracs()``.
    """
    super_lattice = superstructure.lattice.matrix
    prim_lattice = np.linalg.inv(supercell.matrix) @ super_lattice

    # The quotient group of primitive lattice translations modulo the supercell
    translation_carts = supercell.translation_pfracs() @ prim_lattice
    translation_sfracs = translation_carts @ np.linalg.inv(super_lattice)

    # debug_quotient_points(translation_carts[:, :2], super_lattice[:2,:2])

    # The quotient group of supercell reciprocal lattice points modulo the
    # primitive reciprocal cell
    gpoint_sfracs = supercell.gpoint_sfracs()

    # debug_quotient_points((gpoint_sfracs @ np.linalg.inv(super_lattice).T)[:, :2], np.linalg.inv(prim_lattice).T[:2,:2])

    return np.array(list(map_with_progress(
        eigenvectors, progress,
        lambda eigenvector: unfold_one(
            translation_sfracs=translation_sfracs,
            translation_deperms=translation_deperms,
            gpoint_sfracs=gpoint_sfracs,
            kpoint_sfrac=kpoint_sfrac,
            eigenvector=eigenvector.reshape((-1, 3)),
        )
    )))

def unfold_one(
        translation_sfracs,
        translation_deperms,
        gpoint_sfracs,
        kpoint_sfrac,
        eigenvector,
):
    """
    :param translation_sfracs: Shape ``(quotient, 3)``, real.
    The quotient space translations (PC lattice modulo super cell),
    in units of the supercell basis vectors.

    :param translation_deperms: Shape ``(quotient, sc_sites)``, integral.
    Permutations such that ``(carts + translation_carts[i])[deperms[i]]``
    is ordered like the original carts. (i.e. when applied to the coordinate
    data, it translates by ``-1 * translation_carts[i]``)

    For any vector of per-site metadata ``values``, ``values[deperms[i]]`` is
    effectively translated by ``translation_carts[i]``.

    :param gpoint_sfracs: Shape ``(quotient, 3)``, real.
    Translations in the reciprocal quotient space, in units of the reciprocal
    lattice basis vectors.

    (SC reciprocal lattice modulo primitive BZ)

    :param kpoint_sfrac: Shape ``(3,)``, real.
    The K point in the SC reciprocal cell at which the eigenvector was computed.

    :param eigenvector: Shape ``(sc_sites, 3)``, complex.
    A normal mode of the supercell. (arbitrary norm)

    :return: Shape ``(quotient,)``, real.
    Probabilities of `eigenvector` projected onto each kpoint
    ``kpoint + qpoints[i]``.
    """

    translation_sfracs = np.array(translation_sfracs)
    translation_deperms = np.array(translation_deperms)
    gpoint_sfracs = np.array(gpoint_sfracs)
    kpoint_sfrac = np.array(kpoint_sfrac)
    eigenvector = np.array(eigenvector)
    sizes = check_arrays(
        translation_sfracs = (translation_sfracs, ['quotient', 3], np.floating),
        translation_deperms = (translation_deperms, ['quotient', 'sc_sites'], np.integer),
        gpoint_sfracs = (gpoint_sfracs, ['quotient', 3], np.floating),
        kpoint_sfrac = (kpoint_sfrac, [3], np.floating),
        eigenvector = (eigenvector, ['sc_sites', 3], np.floating),
    )

    inner_prods = np.array([
        np.vdot(eigenvector, eigenvector[deperm])
        for deperm in translation_deperms
    ])

    gpoint_probs = []
    for g in gpoint_sfracs:
        # SBZ kpoint dot r for every r
        k_dot_rs = (kpoint_sfrac + g) @ translation_sfracs.T
        phases = np.exp(-2j * np.pi * k_dot_rs)

        prob = sum(inner_prods * phases) / sizes['quotient']

        # analytically, these are all real, positive numbers
        assert abs(prob.imag) < 1e-7
        assert -1e-7 < prob.real
        gpoint_probs.append(max(prob.real, 0.0))
    gpoint_probs = np.array(gpoint_probs)

    np.testing.assert_allclose(gpoint_probs.sum(), np.linalg.norm(eigenvector)**2, atol=1e-7)
    return gpoint_probs

def get_translation_deperm(
        structure: Structure,
        translation_cart,
        axis_mask = np.array([1, 1, 1]),
        tol: float = DEFAULT_TOL,
):
    # NOTE: Heavily-optimized function for identifying permuted structures.
    #       Despite the name, it works just as well for translations as it does
    #       for rotations.
    # FIXME: Shouldn't be relying on this
    from phonopy.structure.cells import compute_permutation_for_rotation

    lattice = structure.lattice.matrix
    fracs_original = structure.frac_coords
    fracs_translated = (structure.cart_coords + translation_cart) @ np.linalg.inv(lattice)

    # Possibly ignore displacements along certain axes.
    fracs_original *= axis_mask
    fracs_translated *= axis_mask

    # Compute the inverse permutation on coordinates, which is the
    # forward permutation on metadata ("deperm").
    #
    # I.e. ``fracs_translated[deperm] ~~ fracs_original``
    return compute_permutation_for_rotation(
        fracs_translated, fracs_original, lattice, tol,
    )

def find_repeats(supercell_matrix):
    """
    Get the number of distinct translations along each lattice primitive
    translation. (it's the diagonal of the row-based HNF of the matrix)

    :param supercell_matrix: Shape ``(3, 3)``, integer.
    Integer matrix satisfying
    ``matrix @ prim_lattice_matrix == super_lattice_matrix``
    where the lattice matrices are understood to store a lattice primitive
    translation in each row.

    :return:
    """
    from abelian import hermite_normal_form
    from sympy import Matrix

    expected_volume = abs(round(np.linalg.det(supercell_matrix)))

    supercell_matrix = Matrix(supercell_matrix) # to sympy

    # abelian.hermite_normal_form is column-based, so give it the transpose
    hnf = hermite_normal_form(supercell_matrix.T)[1].T
    hnf = np.array(hnf).astype(int) # to numpy

    assert round(np.linalg.det(hnf)) == expected_volume
    return np.diag(hnf)

def griddata_periodic(
        points,
        values,
        xi,
        lattice,
        # set this to reduce memory overhead
        periodic_axis_mask=(1,1,1),
        **kwargs,
):
    """
    scipy.interpolate.griddata, but where points (in cartesian) are periodic
    with the given lattice.  The data provided is complemented by images
    from the surrounding unit cells.

    The lattice is assumed to have small skew.
    """
    points_frac = reduce_carts(points, lattice) @ np.linalg.inv(lattice)
    xi_frac = reduce_carts(xi, lattice) @ np.linalg.inv(lattice)
    #debug_path((points_frac @ lattice)[:,:2], lattice[:2,:2], (xi_frac @ lattice)[:,:2])

    for axis, mask_bit in enumerate(periodic_axis_mask):
        if mask_bit:
            unit = [0] * 3
            unit[axis] = 1

            points_frac = np.vstack([
                points_frac - unit,
                points_frac,
                points_frac + unit,
                ])
            values = np.hstack([values] * 3)

    points = points_frac @ lattice
    xi = xi_frac @ lattice

    # Delete axes in which the points have no actual extent, because
    # they'll make QHull mad. (we'd be giving it a degenerate problem)
    true_axis_mask = [1, 1, 1]
    for axis in reversed(range(3)):
        max_point = points_frac[:, axis].max()
        if np.allclose(max_point, points_frac[:, axis].min()):
            np.testing.assert_allclose(max_point, xi_frac[:, axis].min())
            np.testing.assert_allclose(max_point, xi_frac[:, axis].max())

            xi = np.delete(xi, axis, axis=1)
            points = np.delete(points, axis, axis=1)
            true_axis_mask[axis] = 0

    #if xi.shape[1] == 2:
    #    debug_path(points, lattice, xi)

    return scint.griddata(points, values, xi, **kwargs)

def truncate(array, tol):
    array = array.copy()
    if sparse.issparse(array):
        data = array.data
        data[np.absolute(data) < tol] = 0.0
        return array
    else:
        array[np.absolute(array) < tol] = 0.0
        return array

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

def product(iter):
    from functools import reduce
    return reduce((lambda a, b: a * b), iter)

#---------------------------------------------------------------
# debugging

def debug_quotient_points(points2, lattice2):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 8))
    draw_unit_cell(ax, lattice2, lw=2)
    draw_reduced_points(ax, points2, lattice2)
    ax.set_aspect('equal', 'box')
    plt.show()

def debug_path(points, lattice, path):
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches

    lattice = lattice[:, :2] # FIXME
    lattice_path = Path(
        vertices=[[0,0], lattice[0,], lattice[0]+lattice[1], lattice[1], [0,0]],
        codes = [Path.MOVETO] + [Path.LINETO] * 4,
    )
    mpl_path = Path(
        vertices=path,
        codes = [Path.MOVETO] + [Path.LINETO] * (len(path) - 1),
    )

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(points[:, 0], points[:, 1])
    ax.add_patch(patches.PathPatch(lattice_path, facecolor='none', lw=2))
    ax.add_patch(patches.PathPatch(mpl_path, facecolor='none', ls=':', lw=1))
    plt.show()

def draw_path(ax, path, **kw):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    path = np.array(path)
    print(path.shape)
    mpl_path = Path(
        vertices=path,
        codes=[Path.MOVETO] + [Path.LINETO] * (len(path) - 1),
    )
    ax.add_patch(patches.PathPatch(mpl_path, facecolor='none', **kw))

def draw_unit_cell(ax, lattice2, **kw):
    np.testing.assert_equal(lattice2.shape, [2,2])
    path = [[0,0], lattice2[0], lattice2[0]+lattice2[1], lattice2[1], [0,0]]
    draw_path(ax, path, **kw)

def draw_reduced_points(ax, points2, lattice2, **kw):
    points2 = reduce_carts(points2, lattice2)
    ax.scatter(points2[:, 0], points2[:, 1], **kw)

def check_arrays(**kw):
    previous_values = {}

    kw = {name: list(data) for name, data in kw.items()}
    for name in kw:
        if not sparse.issparse(kw[name][0]):
            kw[name][0] = np.array(kw[name][0])

    for name, data in kw.items():
        if len(data) == 2:
            array, dims = data
            dtype = None
        elif len(data) == 3:
            array, dims, dtype = data
        else:
            raise TypeError(f'{name}: Expected (array, shape) or (array, shape, dtype)')

        if dtype and not issubclass(np.dtype(array.dtype).type, dtype):
            raise TypeError(f'{name}: Expected data type {dtype}, got {array.dtype}')

        # format names without quotes
        nice_expected = '[' + ', '.join(map(str, dims)) + ']'
        if len(dims) != array.ndim:
            raise TypeError(f'{name}: Wrong number of dimensions (expected shape {nice_expected}, got {list(array.shape)})')

        for axis, dim in enumerate(dims):
            if isinstance(dim, int):
                if array.shape[axis] != dim:
                    raise TypeError(f'{name}: Mismatched dimension (expected shape {nice_expected}, got {list(array.shape)})')
            elif isinstance(dim, str):
                if dim not in previous_values:
                    previous_values[dim] = (array.shape[axis], name, axis)

                if previous_values[dim][0] != array.shape[axis]:
                    prev_value, prev_name, prev_axis = previous_values[dim]
                    raise TypeError(
                        f'Conflicting values for dimension {repr(dim)}:\n'
                        f' {prev_name}: {kw[prev_name][0].shape} (axis {prev_axis}: {prev_value})\n'
                        f' {name}: {array.shape} (axis {axis}: {array.shape[axis]})'
                    )

    return {dim:tup[0] for (dim, tup) in previous_values.items()}

#---------------------------------------------------------------
# utils

def map_with_progress(
        xs: tp.Iterator[A],
        progress: tp.Callable[[int, int], None],
        function: tp.Callable[[A], B],
) -> tp.Iterator[B]:
    yield from (function(x) for x in iter_with_progress(xs, progress))

def iter_with_progress(
        xs: tp.Iterator[A],
        progress: tp.Callable[[int, int], None],
) -> tp.Iterator[A]:
    xs = list(xs)

    for (num_done, x) in enumerate(xs):
        if progress:
            progress(num_done, len(xs))

        yield x

    if progress:
        progress(len(xs), len(xs))


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def die(*args, **kw):
    print('unfold:', *args, **kw, file=sys.stderr)
    if SHOW_ACTION_STACK:
        for name in ACTION_STACK[::-1]:
            print(f"  while computing {name}", file=sys.stderr)
    sys.exit(1)

#---------------------------------------------------------------

if __name__ == '__main__':
    main()
