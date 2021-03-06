#!/usr/bin/env python3

# converts a POSCAR into a lammps data file that is suitable for use with ./lmp.in
#
# it is suggested that you not use this directly, but rather use
# `./make-expected` to automate the process of running lammps.
# (that script also outputs the json file for expected output)

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
    yx = np.dot(b,a)/norm(a)
    yy = (np.dot(b,b) - yx*yx)**0.5
    zx = np.dot(c,a)/norm(a)
    zy = (np.dot(b,c) - yx*zx)/yy
    zz = (np.dot(c,c) - zx*zx - zy*zy)**0.5

    if is_negative:
        xx = -xx

    return np.array([[xx, 0, 0], [yx, yy, 0], [zx, zy, zz]])

def lammps_friendly(structure):
    orig_lattice = structure.lattice.matrix
    lmp_lattice = lammps_lattice(orig_lattice)
    fracs = structure.frac_coords
    return Structure(lmp_lattice, structure.species, fracs)

input = sys.argv[1]
if input.endswith('.xz'):
    import lzma
    with lzma.open(input, 'rt') as f:
        poscar_str = f.read()
else:
    with open(input, 'rt') as f:
        poscar_str = f.read()
structure = Poscar.from_string(poscar_str).structure
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
print(f'{lattice[1][0]} {lattice[2][0]} {lattice[2][1]} xy xz yz')
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
