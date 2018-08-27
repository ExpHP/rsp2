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

import numpy as np
import json
import sys

from . import eigsh_custom, get_OPinv
from rsp2.internals import info
from rsp2.io import dynmat, eigensols

# The default tolerance for eigsh is machine precision, which I feel is
# overkill. Hopefully a lighter tolerance will save some time.
#
# TODO maybe: experiment with this
TOL = 1e-10

# absolute cosines greater than this are deemed non-orthogonal
OVERLAP_THRESH = 1e-6

def main(d):
    m = dynmat.from_dict(d.pop('matrix'))
    shift_invert_attempts = d.pop('shift-invert-attempts')
    assert not d

    if shift_invert_attempts:
        esols = try_shift_invert(m, shift_invert_attempts)
        if not all(acousticness(v) > 1. - 1e-3 for v in esols[1]):
            return esols

    return try_regular(m)

# As an optimization, begin by using shift-invert mode, which can converge
# in **significantly** fewer iterations than regular mode.
# noinspection PyUnreachableCode
def try_shift_invert(m, shift_invert_attempts):
    info('trace: precomputing OPinv for shift-invert')

    # From what I have seen, shift_invert mode tends to find most of its
    # solutions fairly quickly, but some may be incorrect. Furthermore, it does
    # not always find all valid negative solutions.
    #
    # I fear that when incorrect solutions appear, it may compromise those found
    # afterwards. So we make multiple calls with a small number of iterations.
    MAX_ITER = max(30, int(10 * m.shape[0] ** (1/3)))

    # This many solutions will almost never converge in our short iteration
    # limit. This mostly just affects the default number of lanczos vectors.
    HOW_MANY_SOLS = min(10, m.shape[0] - 2)

    # A heavy computational step at the beginning of shift-invert mode is
    # factorizing the matrix; do that ahead of time.
    OPinv = get_OPinv(m, sigma=0, tol=TOL)

    found_evals = []
    found_evecs = []

    # debug info
    counts = []

    for call_i in range(shift_invert_attempts):
        info('trace: shift-invert call', call_i + 1)
        (evals, evecs) = eigsh_custom(
            m,
            k=HOW_MANY_SOLS,
            maxiter=MAX_ITER,
            sigma=0,
            which='SA',
            tol=TOL,
            OPinv=OPinv,
            allow_fewer_solutions=True,
            # TODO: play around with ncv.
            #       Larger ncv is slower, but what do we get in return?
        )
        evecs = np.array(list(map(normalize, evecs)))

        # a tree of counts based on direct field assignment so that static
        # linters can catch typos. (CLion handles it very impressively!)
        class Count:
            def total(self): return int(self)
            def __int__(self): return sum(map(int, self.__dict__.values()))

        count = Count() # total solutions found
        count.good = 0 # total solutions kept
        count.bad = Count() # total solutions rejected
        count.bad.repeat = 0 # linearly dependent with prior solutions
        count.bad.wrong = 0  # non-eigenvector solutions
        count.bad.ortho_bad = 0 # tried to orthogonalize, got a non-eigenvector
        count.bad.ortho_fail = 0 # tried to orthogonalize, and failed

        for (eval, ev) in zip(evals, evecs):
            # Is it ACTUALLY an eigenvector?
            if not is_good_esol(m, eval, ev):
                count.bad.wrong += 1
                continue

            # Linearly dependent with existing solutions?
            if sum(np.abs(np.vdot(ev, other))**2 for other in found_evecs) > 0.95:
                count.bad.repeat += 1
                continue

            # Prepare it for possible insertion.
            ortho_ev = mgs_step(ev, found_evecs)

            # We didn't ruin it, did we?
            if not is_good_esol(m, eval, ortho_ev):
                count.bad.ortho_bad += 1
                continue

            if sum(np.abs(np.vdot(ortho_ev, other))**2 for other in found_evecs) > 1e-6:
                count.bad.ortho_fail += 1
                continue

            # ship it
            count.good += 1
            found_evecs.append(ortho_ev)
            found_evals.append(eval)

        counts.append(count)

    info(" Good -- Bad (Old Wrong OrthoFail OrthoBad)")
    for count in counts:
        info(
            " {:^4} -- {:^3} ({:^3} {:^5} {:^9} {:^8})".format(
                count.good,
                count.bad.total(),
                count.bad.repeat,
                count.bad.wrong,
                count.bad.ortho_fail,
                count.bad.ortho_bad,
            )
        )

    perm = np.argsort(found_evals)
    evals = np.array(found_evals)[perm]
    evecs = np.array(found_evecs)[perm]
    for val, v in zip(evals, evecs):
        if not is_good_esol(m, val, v):
            np.save('bad-mat.npy', m)
            np.save('bad-vec.npy', v)
            assert False, "bad evec"
    for i in range(len(evecs)):
        for j in range(i):
            if is_overlapping(evecs[i], evecs[j]):
                np.save('bad-a.npy', evecs[i])
                np.save('bad-b.npy', evecs[j])
                assert False, "overlap"
    return evals, evecs

def mgs_step(a, b_hats):
    """
    This is the function such that

    >>> def mgs(original_vecs):
    ...     out = []
    ...     for vec in original_vecs:
    ...         out.append(mgs_step(vec, out))
    ...     return out

    is a correct implementation of Modified Gram Schmidt method.

    Many sources present the Modified Gram Schmidt (MGS) method in a misleading
    light by presenting it as having a different order of iteration from the
    famously unstable Classical Gram-Schmidt (CGS) method, and suggesting that
    the change in iteration order is the cause for MGS' improved numerical
    stability.  More specifically, MGS is typically presented as taking each
    vector in turn and using it to modify the vectors that come **after** it,
    in contrast to CGS which modifies each vector based on those **prior** to
    it. This could lead a naive reader to believe that the function `mgs_step`
    cannot exist!

    This is a complete ruse, however. The change in iteration order does not
    change the dependency tree of floating point operations. (it merely
    reintroduces opportunities for parallelism that would otherwise be lost
    compared to CGS)

    The ACTUAL difference between CGS and MGS is best understood by contrasting
    their step functions. Here's what `cgs_step` would look like. Notice how
    all dot products involve the original vector:

    >>> def cgs_step(a, b_hats):
    >>>     original = a.copy()
    >>>     for b_hat in b_hats:
    >>>         a = a - par(original, b_hat)
    >>>     return normalize(a)

    Contrast that with the following:
    """

    # Yes, for all of that text, *it really is this simple.*
    for b_hat in b_hats:
        a = a - par(a, b_hat)
    return normalize(a)

# The part of `a` that points along `b_hat`.
def par(a, b_hat):
    return np.vdot(b_hat, a) * b_hat

def acousticness(v_hat):
    sum = np.reshape(v_hat, (-1, 3)).sum(axis=0)
    return abs(np.vdot(sum, sum))

def normalize(v):
    return v / np.sqrt(np.vdot(v, v))

def is_overlapping(a_hat, b_hat):
    return abs(np.vdot(a_hat, b_hat)) > OVERLAP_THRESH

def is_good_esol(m, eval, evec):
    assert abs(abs(np.vdot(evec, evec)) - 1) < 1e-12
    return lazy_any([
        lambda: acousticness(evec) > 1. - 1e-3,
        lambda: lazy_all([
            lambda: abs(abs(np.vdot(normalize(m @ evec), evec)) - 1.0) < 1e-2,
            lambda: (np.abs(m @ evec - eval * evec) < TOL * 10).all(),
        ])
    ])

def lazy_any(it): return any(pred() for pred in it)
def lazy_all(it): return all(pred() for pred in it)

# If shift-invert hasn't produced anything satisfactory, try regular mode.
# From what I've seen, this always produces legitimate solutions, but generally
# takes long to converge onto anything.
def try_regular(m):
    info('trace: trying non-shift-invert')
    return eigsh_custom(
        m,
        k=min(12, m.shape[0]-1),
        which='SA',
        tol=TOL,
        allow_fewer_solutions=True,
        # TODO: play around with ncv.
        #       Larger ncv is slower, but what do we get in return?
    )

if __name__ == '__main__':
    json.dump(eigensols.to_cereal(main(json.load(sys.stdin))), sys.stdout)
    print(file=sys.stdout)
