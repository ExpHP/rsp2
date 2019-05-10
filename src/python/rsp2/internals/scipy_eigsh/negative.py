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
import scipy.sparse
import typing as tp

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

DEFAULT_MAX_SOLUTIONS = 12
DEFAULT_SHIFT_INVERT_ATTEMPTS = 4
DEFAULT_NCV = 0

def main_from_rust():
    """
    Entry point when called from rsp2's rust code.

    Communicates through JSON over the standard IO streams.
    """
    info('trace: sending dynmat from rust to python')
    d = json.load(sys.stdin)
    m = dynmat.from_dict(d.pop('matrix'))
    shift_invert_attempts = d.pop('shift-invert-attempts')
    dense = d.pop('dense')
    max_solutions = d.pop('max-solutions')
    assert not d

    out = run(m,
              dense=dense,
              shift_invert_attempts=shift_invert_attempts,
              plain_ncv=DEFAULT_NCV, # FIXME add to input json from rust
              shift_invert_ncv=DEFAULT_NCV, # FIXME add to input json from rust
              use_fallback=True, # FIXME add to input json from rust
              max_solutions=max_solutions,
              search_solutions=None, # FIXME add to input json from rust
              )

    info('trace: sending eigensolutions from python to rust')
    json.dump(eigensols.to_cereal(out), sys.stdout)
    print(file=sys.stdout) # newline

def main_from_cli():
    """
    Entry point for the standalone CLI wrapper.
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('DYNMAT', help='dynmat file (.npz, .json, .json.gz, ...)')
    p.add_argument('--output', '-o', type=str, required=True)
    p.add_argument(
        '--dense', action='store_true',
        help="Use a dense eigenvalue solver. Almost all other options will be "
             "ignored in this case."
    )
    p.add_argument('--shift-invert-attempts', type=int, default=DEFAULT_SHIFT_INVERT_ATTEMPTS)
    p.add_argument(
        '--no-fallback', dest='use_fallback', action='store_false',
        help="Disable non-shift-invert-based fallback method."
    )
    p.add_argument(
        '--max-solutions', type=int, default=None,
        help="max number of solutions to seek. Default is 12 unless --dense is given."
    )
    p.add_argument(
        '--shift-invert-ncv', type=int, default=DEFAULT_NCV,
        help="suggested number of Lanczos vectors. This will automatically be "
             "clipped into the range of [min(2*max_solutions + 1, n), n]"
    )
    p.add_argument(
        '--plain-ncv', type=int, default=DEFAULT_NCV,
        help="suggested number of Lanczos vectors. This will automatically be "
             "clipped into the range of [min(2*max_solutions + 1, n), n]"
    )
    p.add_argument(
        '--search-solutions', type=int, default=None,
        help="actually ask the sparse solver for this many solutions instead "
             "of --max-solutions.  The sparse solver can converge much, much "
             "faster when a few hundred solutions are requested rather than "
             "just 12."
    )
    args = p.parse_args()

    if (not args.dense
            and args.search_solutions is not None
            and args.max_solutions is not None
            and args.search_solutions < args.max_solutions
    ):
        p.error("--max-solutions must not exceed --search-solutions")

    out = run(
        m=dynmat.from_path(args.DYNMAT),
        dense=args.dense,
        shift_invert_attempts=args.shift_invert_attempts,
        shift_invert_ncv=args.shift_invert_ncv,
        plain_ncv=args.plain_ncv,
        max_solutions=args.max_solutions,
        use_fallback=args.use_fallback,
        search_solutions=args.search_solutions,
    )
    eigensols.to_path(args.output, out)

def run(m: scipy.sparse.bsr_matrix,
        dense: bool,
        shift_invert_attempts: int,
        shift_invert_ncv: int,
        plain_ncv: int,
        max_solutions: tp.Optional[int],
        use_fallback: bool,
        search_solutions: tp.Optional[int],
        ):
    """
    A suitable entry point from pure python code.
    """
    if max_solutions is None:
        if dense:
            max_solutions = m.shape[0]
        else:
            max_solutions = DEFAULT_MAX_SOLUTIONS

    if search_solutions is None:
        search_solutions = max_solutions

    # Logic for deciding when to use shift invert results
    def inner():
        if dense:
            return try_dense(m, max_solutions)

        if shift_invert_attempts:
            esols = try_shift_invert(m,
                                     max_solutions=search_solutions,
                                     shift_invert_attempts=shift_invert_attempts,
                                     ncv=shift_invert_ncv,
                                     )
            if not all(acousticness(v) > 1. - 1e-3 for v in esols[1]):
                return esols

        if use_fallback:
            return try_regular(m,
                               max_solutions=search_solutions,
                               ncv=plain_ncv,
                               )
        else:
            raise RuntimeError('Failed to diagonalize matrix!')

    # Cutting out other solutions
    evals, evecs = inner()
    if not len(evals):
        raise RuntimeError('No solutions found!')
    return evals[:max_solutions], evecs[:max_solutions]

# As an optimization, begin by using shift-invert mode, which can converge
# in **significantly** fewer iterations than regular mode.
# noinspection PyUnreachableCode
def try_shift_invert(m, *, shift_invert_attempts, max_solutions, ncv):
    info('trace: precomputing OPinv for shift-invert')

    # From what I have seen, shift_invert mode tends to find most of its
    # solutions fairly quickly, but some may be incorrect. Furthermore, it does
    # not always find all valid negative solutions.
    #
    # I fear that when incorrect solutions appear, it may compromise those found
    # afterwards. So we make multiple calls with a small number of iterations.
    MAX_ITER = max(30, int(10 * m.shape[0] ** (1/3)))

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
            k=max_solutions,
            maxiter=MAX_ITER,
            sigma=0,
            which='SA',
            tol=TOL,
            OPinv=OPinv,
            ncv=ncv,
            allow_fewer_solutions=True,
            auto_adjust_k=True,
            auto_adjust_ncv=True,
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
    ...     original = a.copy()
    ...     for b_hat in b_hats:
    ...         a = a - par(original, b_hat)
    ...     return normalize(a)

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
def try_regular(m, *, max_solutions, ncv):
    info('trace: trying non-shift-invert')

    return eigsh_custom(
        m,
        k=max_solutions,
        which='SA',
        tol=TOL,
        ncv=ncv,
        allow_fewer_solutions=True,
        auto_adjust_k=True,
        auto_adjust_ncv=True,
    )

def try_dense(m, max_solutions: int):
    info('trace: using dense eigensolver')

    # note: order for returned eigenvalues is the same as 'SA'
    # note: raises LinAlgError if the eigenvalue computation does not converge
    evals, evecs = np.linalg.eigh(m.todense())
    return evals[:max_solutions], evecs.T[:max_solutions]

if __name__ == '__main__':
    main_from_rust()
