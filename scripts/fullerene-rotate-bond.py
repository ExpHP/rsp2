#!/usr/bin/env python3

import numpy as np
import argparse
import os
import sys

PROG = os.path.basename(sys.argv[0])

def main():
    parser = argparse.ArgumentParser(
        description='Rotate a bond (Stone-Wales defect) of a fullerene stored in a POSCAR file.',
    )
    parser.add_argument('INPUT', help='POSCAR where all atoms use an image from the same molecule.')
    parser.add_argument('ATOM1', type=int, help='0-based index of first bonded atom')
    parser.add_argument('ATOM2', type=int, help='0-based index of second bonded atom')
    args = parser.parse_args()

    from pymatgen.io.vasp import Poscar
    from pymatgen.core import Structure
    poscar = Poscar.from_file(args.INPUT)
    structure = poscar.structure

    cart_coords = structure.cart_coords
    cart_coords = rotate_bond(cart_coords, args.ATOM1, args.ATOM2)
    poscar.structure = Structure(structure.lattice, structure.species, cart_coords, coords_are_cartesian=True)

    print(poscar.get_string(significant_figures=16), end='')

def rotate_bond(cart_coords, atom1, atom2):
    cart_coords = cart_coords.copy()

    mol_center = np.mean(cart_coords, axis=0) # center of mass (unweighted; all Carbon)
    bond_coords = cart_coords[[atom1, atom2]]
    bond_center = np.mean(bond_coords, axis=0)
    axis = bond_center - mol_center

    bond_rs = bond_coords - bond_center

    # under the similarity transform that rotates the bond center to the +z axis,
    # rotate the atoms 90 degrees around z.
    twist_bond = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    unitary_transform = rotate_to_z_axis(axis)
    # (outer .T is because we're transforming rows)
    new_bond_rs = bond_rs @ (unitary_transform.T @ twist_bond @ unitary_transform).T
    new_bond_coords = new_bond_rs + bond_center

    cart_coords[[atom1, atom2]] = new_bond_coords
    return cart_coords

def normalize(arr):
    return np.array(arr) / np.linalg.norm(arr)

def rotate_to_z_axis(pos):
    ρ = np.linalg.norm(pos[:2])
    r = np.linalg.norm(pos)

    transform = np.eye(3)
    # rotate pos to xz plane
    if np.any(np.absolute(pos[:2]) > 1e-8):
        cφ, sφ = normalize(pos[:2])
        transform = np.array([[cφ, sφ, 0], [-sφ, cφ, 0], [0, 0, 1]]) @ transform
    assert np.absolute((transform @ pos)[1] < 1e-8)
    # rotate pos to x axis
    cθ, sθ = normalize([ρ, pos[2]])
    transform = np.array([[cθ, 0, sθ], [0, 1, 0], [-sθ, 0, cθ]]) @ transform
    # permute xyz axes
    transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ transform

    transformed = transform @ pos
    assert np.all(np.absolute(transformed)[:2] < 1e-8), transformed
    assert transformed[2] > 0, transformed
    return transform

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
