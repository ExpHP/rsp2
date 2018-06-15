#!/usr/bin/env python3

###########################################################################
#  This file is part of rsp2, and is licensed under EITHER the MIT license
#  or the Apache 2.0 license, at your option.
#
#      http://www.apache.org/licenses/LICENSE-2.0
#      http://opensource.org/licenses/MIT
#
#  Be aware that not all of rsp2 is provided under this permissive license,
#  and that the project as a whole is licensed under the GPL 3.0.
###########################################################################

import json
import sys
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence

import time

# (you can add calls to this if you want to do some "println profiling".
#  rsp2 will timestamp the lines as they are received)
def info(s):
    print(s, file=sys.stderr); sys.stderr.flush(); time.sleep(0)

d = json.load(sys.stdin)
allow_non_convergence = d.pop('allow-fewer-solutions')
kw = d.pop('kw')
m = d.pop('matrix')
assert not d

data = np.array(m['complex-blocks'])

assert data.shape[1] == 2

# scipy will use eigs for dtype=complex even if imag is all zero.
# Unfortunately this leads to small but nonzero imaginary components
# appearing in the output eigenkets at gamma.
# Hence we force dtype=float when possible.
if np.absolute(data[:, 1]).max() == 0:
    data = data[:, 0] * 1.0
else:
    data = data[:, 0] + 1.0j * data[:, 1]

assert data.ndim == 3
assert data.shape[1] == data.shape[2] == 3
m = scipy.sparse.bsr_matrix(
    (data, m['col'], m['row-ptr']),
    shape=tuple(3*x for x in m['dim']),
).tocsc()

try:
    (vals, vecs) = scipy.sparse.linalg.eigsh(m, **kw)

# This can quite easily happen if, say, we use shift-invert mode with
# which='SA' (most negative), and request more eigenvalues than the
# amount that actually exists with eigenvalue < sigma.
except ArpackNoConvergence as e:
    if allow_non_convergence:
        vals = e.eigenvalues
        vecs = e.eigenvectors
    else:
        raise

real = vecs.real.T.tolist()
imag = vecs.imag.T.tolist()
vals = vals.tolist()
json.dump((vals, (real, imag)), sys.stdout)
print()
