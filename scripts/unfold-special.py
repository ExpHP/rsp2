#!/usr/bin/env python3

import os
import sys
import math
import argparse
import numpy as np
import typing as tp

PROG = os.path.basename(sys.argv[0])

try:
    import rsp2
    import unfold_lib
    import unfold
    from rsp2.io import dwim
    from unfold_lib import coalesce
except ImportError:
    info = lambda s: print(s, file=sys.stderr)
    info('Please add the following to your PYTHONPATH:')
    info('  (rsp2 source root)/scripts')
    info('  (rsp2 source root)/src/python')
    info("Or alternatively, create a conda environment from rsp2's environment.yml")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description=
        'A more limited form of unfold.py that extracts band data unfolded onto '
        "just one or more specific kpoints (using some of unfold.py's intermediate "
        "output files as input).",
    )
    parser.add_argument(
        'STRUCTURE', help=
        'Path to initial structure in rsp2 directory structure format.',
    )
    parser.add_argument(
        '--multi-qpoint-file', required=True, help=
        'Path to multi-qpoint.yaml file (see the unfold.py script for more details)',
    )

    parser.add_argument(
        '--coalesce-threshold', type=float,
        metavar='THRESHOLD', help=
        'Coalesce modes of similar eigenvalue into one for reduced output size. '
        'When using this, some mode probabilities may exceed 1.'
    )
    parser.add_argument(
        '--probs-threshold', type=float, default=1e-7, help=
        'Truncate probabilities smaller than this when writing output. '
    )

    v2_to_v3 = lambda v2: np.array(list(v2) + [0])
    jorio_radians = lambda rad: v2_to_v3(jorio_rband_pfrac(rad))
    jorio_degrees = lambda deg: parse_jorio_radians(math.radians(deg))
    parse_jorio_radians = lambda s: jorio_radians(float(s))
    parse_jorio_degrees = lambda s: jorio_degrees(float(s))
    parser.set_defaults(kpoints=[])
    parser.add_argument(
        '--jorio-degrees',
        action='append', dest='kpoints', type=parse_jorio_degrees, help=
        "Unfold onto Jorio's q(theta), for the specified angle in degrees.",
    )
    parser.add_argument(
        '--jorio-radians',
        action='append', dest='kpoints', type=parse_jorio_radians, help=
        "Unfold onto Jorio's q(theta), for the specified angle in radians.",
    )
    parser.add_argument(
        '--kpoint',
        action='append', dest='kpoints', type=unfold.TaskQpointSfrac.parse, help=
        'Unfold onto this K-point, in fractional coordinates of the '
        'primitive reciprocal cell, as a whitespace-separated list of '
        '3 floats, or rational numbers.'
    )

    parser.set_defaults(output=[])
    parser.add_argument(
        '--output', action='append', help=
        'Output file for a single kpoint. Must be specified once for each '
        "option that defines a kpoint. (any format supported by rsp2's DWIM"
        'IO mechanisms, e.g. .json.xz)',
    )
    args = parser.parse_args()

    if not args.kpoints:
        parser.error('No points specified to unfold onto!')

    if len(args.kpoints) != len(args.output):
        parser.error('Number of kpoints to unfold onto must match number of output files!')

    sdir = rsp2.io.structure_dir.from_path(args.STRUCTURE)
    multi_qpoint_data = unfold.TaskMultiQpointData.read_file(args.multi_qpoint_file, probs_threshold=1e-7)
    main_(
        sdir=sdir,
        multi_qpoint_data=multi_qpoint_data,
        kpoints_pfrac=np.array(args.kpoints),
        output_paths=args.output,
        coalesce_threshold=args.coalesce_threshold,
        probs_threshold=args.probs_threshold,
    )

def main_(
        sdir: rsp2.io.structure_dir.StructureDir,
        multi_qpoint_data: dict,
        kpoints_pfrac: np.ndarray,
        output_paths: tp.List[str],
        coalesce_threshold: tp.Optional[float],
        probs_threshold: float,
):
    sc_matrix = sdir.layer_sc_matrices[0]
    supercell = unfold_lib.Supercell(sc_matrix)
    super_lattice = sdir.structure.lattice.matrix
    prim_lattice = np.linalg.inv(sc_matrix) @ super_lattice

    super_recip_lattice = np.linalg.inv(super_lattice).T
    prim_recip_lattice = np.linalg.inv(prim_lattice).T

    kpoints_pfrac %= 1

    resampled = unfold.resample_qg_indices(
        super_lattice=super_lattice,
        supercell=supercell,
        qpoint_sfrac=multi_qpoint_data['qpoint-sfrac'],
        path_kpoint_pfracs=np.array(kpoints_pfrac),
    )
    resampled_qs = resampled['Q']
    resampled_gs = resampled['G']

    closest_images_sfrac = np.array(multi_qpoint_data['qpoint-sfrac'])[resampled_qs] + supercell.gpoint_sfracs()[resampled_gs]
    closest_images_cart = closest_images_sfrac @ super_recip_lattice
    closest_images_cart = unfold_lib.reduce_carts(closest_images_cart, prim_recip_lattice)

    kpoints_cart = kpoints_pfrac @ prim_recip_lattice

    # The point we actually unfolded onto might not be the point we wanted.
    # Check how far the two points are from each other and record it.
    errors_cart = np.array([
        unfold_lib.shortest_image_norm(diff, prim_recip_lattice)
        for diff in kpoints_cart - closest_images_cart
    ])

    for i in range(len(resampled_qs)):
        resampled_q = resampled_qs[i]
        resampled_g = resampled_gs[i]
        error_distance = np.linalg.norm(errors_cart[i])

        ev_frequencies = multi_qpoint_data['mode-data']['ev_frequencies'][resampled_q]
        ev_probs = multi_qpoint_data['probs'][resampled_q].T[resampled_g]
        ev_probs = np.asarray(ev_probs.todense()).squeeze()

        if coalesce_threshold is not None:
            splits = list(unfold_lib.coalesce.get_splits(ev_frequencies, coalesce_threshold))
            ev_probs = unfold_lib.coalesce.coalesce(splits, ev_probs, 'sum')
            ev_probs = ev_probs[splits[:-1]]
            ev_frequencies = unfold_lib.coalesce.coalesce(splits, ev_frequencies, 'mean')
            ev_frequencies = ev_frequencies[splits[:-1]]

            mask = ev_probs > probs_threshold
            ev_probs = ev_probs[mask]
            ev_frequencies = ev_frequencies[mask]

        dwim.to_path(output_paths[i], {
            'sample-error-distance': error_distance,
            'ev-probs': ev_probs.tolist(),
            'ev-frequencies': ev_frequencies.tolist(),
        })

# ------------------------------------------------------

def rotation_matrix_22(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])

def rotation_matrix_33(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def jorio_rband_pfrac(twist_angle):
    jorio_lattice = np.array([[1, 0], [-0.5, 0.5*3**0.5]]) @ rotation_matrix_22(np.radians(90)).T

    cos, sin = math.cos(twist_angle), math.sin(twist_angle)
    q_jorio_cart = (3**-0.5) * np.array([
        -(1 - cos) - 3**0.5 * sin,
        -3**0.5 * (1 - cos) + sin,
    ])

    q_pfrac = q_jorio_cart @ jorio_lattice.T
    return q_pfrac

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
