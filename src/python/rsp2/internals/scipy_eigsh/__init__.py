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

# Returns column vectors as ROWS,
# and has some other knobs too
def eigsh_custom(A,
                 *args,
                 # Don't throw an exception when fewer solutions converge
                 # than requested. (instead, return the solutions found)
                 allow_fewer_solutions=False,

                 # Always apply the recommended minimum of NCV=2*k+1,
                 # and clip to the hard maximum of N
                 auto_adjust_ncv=False,

                 # Clip k to the hard maximum of N-1.
                 #
                 # Defaults to match the setting of `allow_fewer_solutions`.
                 auto_adjust_k=None,
                 **kw):

    if auto_adjust_k is None:
        auto_adjust_k = allow_fewer_solutions

    if 'k' not in kw:
        kw['k'] = 6 # Scipy's own default

    if auto_adjust_k:
        kw['k'] = min(kw['k'], A.shape[0] - 1)

    if auto_adjust_ncv and 'ncv' in kw:
        kw['ncv'] = max(kw['ncv'], 2 * kw['k'] + 1) # Recommendation
        kw['ncv'] = min(kw['ncv'], A.shape[0]) # Hard limit

    try:
        evals, evecs = scipy.sparse.linalg.eigsh(A, *args, **kw)
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
    try:
        matvec = spla.eigen.arpack.get_OPinv_matvec(A, M=None, symmetric=True, sigma=sigma, tol=tol)
    except TypeError:
        # argument name changed
        matvec = spla.eigen.arpack.get_OPinv_matvec(A, M=None, hermitian=True, sigma=sigma, tol=tol)
    return spla.LinearOperator(A.shape, matvec=matvec, dtype=A.dtype)
