#!/usr/bin/env python3

import sys
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
from numpy.linalg import norm
import numpy as np

def lammps_lattice(matrix):
    a,b,c = matrix
    is_negative = np.linalg.det(matrix) < 0
    if is_negative:
        a = -a

    xx = norm(a)
    a_unit = a/norm(a)
    yx = np.dot(b, a_unit)
    yy = norm(np.cross(a_unit, b))
    ab_unit = np.cross(a, b) / norm(np.cross(a, b))
    zx = np.dot(c, a_unit)
    zy = np.dot(c, np.cross(ab_unit, a_unit))
    zz = norm(np.dot(c, ab_unit))

    if is_negative:
        xx = -xx

    return np.array([[xx, 0, 0], [yx, yy, 0], [zx, zy, zz]])

def lammps_friendly(structure):
    orig_lattice = structure.lattice.matrix
    lmp_lattice = lammps_lattice(orig_lattice)
    fracs = structure.frac_coords
    return Structure(lmp_lattice, structure.species, fracs)

structure = Poscar.from_file(sys.argv[1]).structure
structure = lammps_friendly(structure)
carts = structure.cart_coords
lattice = structure.lattice.matrix
species = structure.species
print()
print()
print(f'{len(carts)} atoms')
print()
print(f'2 atom types')
print()
print(f'0.0 {lattice[0][0]} xlo xhi')
print(f'0.0 {lattice[1][1]} ylo yhi')
print(f'0.0 {lattice[2][2]} zlo zhi')
print(f'{lattice[0][1]} {lattice[0][2]} {lattice[1][2]} xy xz yz')
print()
print(f'Masses')
print()
print(f'1 12.01')
print(f'2 1.00794')
print()
print('Atoms')
print()
for n, ((x, y, z), s) in enumerate(zip(carts, species), start=1):
    ty = {
        'C': 1,
        'H': 2,
    }[str(s)]
    print(f'{n} {ty} {x} {y} {z}')
