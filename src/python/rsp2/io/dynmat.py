import numpy as np

import scipy.sparse

from . import dwim

# Dynamical matrix.
#
# Multiple formats are accepted:
#
# * a `.npz` file of a complex `scipy.sparse.bsr_matrix`.
# * any general serialization format (via the `dwim` module),
#   where the serialized object is as follows:
#
#   - define constants as follows:
#     * BNNZ: Number of stored 3x3 blocks
#     * BH: Number of rows in block format (true number of rows // 3)
#     * BW: Number of columns in block format (true number of cols // 3)
#
#   - value is a dict with:
#     - 'dim' is the list [BH, BW]
#     - 'col' is a list of BNNZ integers
#     - 'row-ptr' is a list of BH + 1 integers
#     - 'complex-blocks' is list of len BNNZ of
#       - list of len 2 (real, imag) of
#         - 3x3 matrices, as lists of lists
#
# The NPZ format is preferred wherever possible due to its compact size, but
# the other format remains available because NPZ files are impossible to work
# with in any language besides Python.

def from_path(path):
    d = dwim.from_path(path)
    if isinstance(d, dict):
        return from_dict(d)
    elif isinstance(d, np.ndarray) or scipy.sparse.issparse(d):
        return d
    else:
        raise TypeError

def to_path(path, obj):
    dwim.to_path(path, obj, to_dict=to_dict)

def _from_npy(m): return m
def _to_npy(m): return m

def to_dict(m):
    assert isinstance(m, scipy.sparse.bsr_matrix)
    assert m.data.shape[1:] == (3,3)
    assert all(x % 3 == 0 for x in m.shape)

    # [real/imag][block][3][3]
    data = np.array([m.data.real, m.data.imag])
    # [block][real/imag][3][3]
    data = np.swapaxes(data, 0, 1)
    return {
        'complex-blocks': data.tolist(),
        'row-ptr': m.indptr.tolist(),
        'col': m.indices.tolist(),
        'dim': tuple(x // 3 for x in m.shape),
    }

def from_dict(m):
    data = np.array(m['complex-blocks'])

    assert data.ndim == 4
    assert data.shape[1:] == (2, 3, 3)

    # scipy will use eigs for dtype=complex even if imag is all zero.
    # Unfortunately this leads to small but nonzero imaginary components
    # appearing in the output eigenkets at gamma.
    # Hence we force dtype=float when possible.
    if np.absolute(data[:, 1]).max() == 0:
        data = data[:, 0] * 1.0
    else:
        data = data[:, 0] + 1.0j * data[:, 1]

    return scipy.sparse.bsr_matrix(
        (data, m['col'], m['row-ptr']),
        shape=tuple(3*x for x in m['dim']),
    )
