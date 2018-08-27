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

import scipy.sparse
import scipy.sparse.linalg as spla
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence

# NOTE: returns column eigenvectors as ROWS
def eigsh_custom(*args, allow_fewer_solutions=False, **kw):
    try:
        evals, evecs = scipy.sparse.linalg.eigsh(*args, **kw)
        return evals, evecs.T

    # This can quite easily happen if, say, we use shift-invert mode with
    # which='SA' (most negative), and request more eigenvalues than the
    # amount that actually exists with eigenvalue < sigma.
    except ArpackNoConvergence as e:
        if allow_fewer_solutions:
            return e.eigenvalues, e.eigenvectors.T
        else:
            raise

# precompute OPinv for faster repeated shift-invert calls
def get_OPinv(A, sigma, tol=0):
    # FIXME usage of scipy implementation detail
    matvec = spla.eigen.arpack.get_OPinv_matvec(A, M=None, symmetric=True, sigma=sigma, tol=tol)
    return spla.LinearOperator(A.shape, matvec=matvec, dtype=A.dtype)
