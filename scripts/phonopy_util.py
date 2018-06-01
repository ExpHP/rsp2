# NOTE: None of this is used by rsp2.
#       It's just some crap used by the author on the python REPL,
#       checked into VCS for "reasons".

import h5py
import os
from ruamel.yaml import YAML
import numpy as np
from pymatgen import Structure
from pymatgen.io.vasp import Poscar

try:
    from phonopy.structure.cells import _compute_permutation_c
except:
    from phonopy.harmonic.force_constants import _compute_permutation_c

import spglib

yaml=YAML(typ='safe')

def gamma_from_band_hdf5(path='band.hdf5'):
    assert os.path.exists(path)
    h = h5py.File(path)
    q = h['path'].value
    freq = h['frequency'].value
    evs = h['eigenvector'].value
    for stuff in zip(q, freq, evs):
        for (q, freq, evs) in zip(*stuff):
            if (q == 0.0).all():
                evs = evs.T
                n = len(evs) // 3
                return freq, evs.reshape(n * 3, n, 3)
    raise RuntimeError('gamma point not found in hdf5 file')

def read_symmetry_yaml(path='symmetry.yaml'):
    sym = yaml.load(open(path))
    rots = np.array([d['rotation'] for d in sym['space_group_operations']])
    trans = np.array([d['translation'] for d in sym['space_group_operations']])
    return rots, trans

def compute_permutation(fracs_from, fracs_to, lattice, symprec):
    return _compute_permutation_c(fracs_from, fracs_to, np.array(lattice).T, symprec)

# horrifying dwim function
def structure(s):
    if isinstance(s, Structure): return s
    elif isinstance(s, Poscar): return s.structure
    elif isinstance(s, str): return Poscar.from_file(s).structure
    else: raise TypeError

class SpglibError(Exception):
    pass

def spglib_dataset(s, symprec=1e-5):
    s = structure(s)
    fracs = s.frac_coords
    lattice = s.lattice.matrix
    types = [x.number for x in s.species]
    ds = spglib.get_symmetry_dataset((lattice, fracs, types), symprec)

    if not ds:
        error = spglib.get_error_message()
        if error == "no error":
            error += ". (or so spglib claims...)"
        raise SpglibError(spglib.get_error_message())

    return ds
