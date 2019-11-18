import numpy as np
import scipy.sparse as sparse

from .util import run_once

@run_once
def __lazy_define_jitted_funcs():
    try:
        import numba
    except ImportError:
        import warnings
        warnings.warn('coalescing without numba installed. This may be slow!')

        class numba:
            @staticmethod
            def jit(*args, **kw):
                return lambda func: func

    def make_global(f):
        globals()[f.__name__] = f

    @make_global
    @numba.jit(nopython=True)
    def __coalesce_inplace_mean(splits, data):
        for start, end in zip(splits[:-1], splits[1:]):
            data[start] = data[start:end].mean()

    @make_global
    @numba.jit(nopython=True)
    def __coalesce_inplace_sum(splits, data):
        for start, end in zip(splits[:-1], splits[1:]):
            data[start] = data[start:end].sum()

    @make_global
    @numba.jit(nopython=True)
    def __coalesce_inplace_max(splits, data):
        for start, end in zip(splits[:-1], splits[1:]):
            data[start] = data[start:end].max()

    @make_global
    @numba.jit(nopython=True)
    def __coalesce_inplace_weighted_mean(splits, data, weights):
        for start, end in zip(splits[:-1], splits[1:]):
            data[start] = np.vdot(data[start:end], weights[start:end]) / weights[start:end].sum()

def get_splits(data, threshold):
    """ Compute split indices for the ``coalesce`` function from a 1D array of
    sorted data.

    The output is a set of indices that partitions the data into segments
    such that each segment contains at least one value, and contains a range of
    values no wider than the threshold. (beyond these properties, the precise
    selection of groupings is left unspecified). The first element is always 0,
    and the last element is always the length of the data.
    """
    yield 0
    data = iter(data)
    start_value = next(data)

    index = 0  # correct end value in case the loop runs 0 iterations
    for index, x in enumerate(data, start=1):
        if start_value + threshold < x:
            yield index
            start_value = x
    yield index + 1 # length of data

NO_FILL = object()
def coalesce(splits, data, mode, fill=NO_FILL):
    """ Partitions ``data`` at the indices defined in ``splits``, and then
    reduces each partition along that axis using some function specified by
    ``mode``.

    ``mode`` is a string which can be ``"mean"``, ``"sum"``, ``"max"``,
    or ``"first"``.
    (arbitrary ufuncs are not supported due to limitations of numba)

    To help avoid creating situations where jagged arrays are needed,
    the output is the same shape as the input. The output of the ufunc is
    written to the first item in each partition; the rest are optionally
    filled with a constant. """
    return coalesce_inplace(splits, data.copy(), mode, fill=fill)

def coalesce_inplace(splits, data, mode, fill=NO_FILL):
    """ Variant of coalesce that reuses ``data`` for the output buffer. """
    splits = np.array(splits, copy=False)

    __lazy_define_jitted_funcs()

    if mode == "sum": __coalesce_inplace_sum(splits, data)
    elif mode == "mean": __coalesce_inplace_mean(splits, data)
    elif mode == "max": __coalesce_inplace_max(splits, data)
    elif mode == "first": pass # the first value of each group is already there!
    else: raise ValueError(f'Unknown reduction mode: {mode}')

    __coalesce_fill(splits, data, fill=fill)

    return data

def coalesce_inplace_weighted_mean(splits, data, weights, fill=NO_FILL):
    """ Variant of ``coalesce_inplace`` that performs a weighted mean. """
    __lazy_define_jitted_funcs()

    __coalesce_inplace_weighted_mean(splits, data, weights, fill)
    __coalesce_fill(splits, data, fill=fill)

    return data

def __coalesce_fill(splits, data, fill=NO_FILL):
    if fill is not NO_FILL:
        inv_mask = np.ones((splits[-1],), dtype=bool)
        inv_mask[splits[:-1]] = False
        data[inv_mask] = fill

def coalesce_sparse_row_vec(splits, csr, mode):
    """ Variant of `coalesce` for 1xN CSR matrices.

    The unused elements of the output are always set to the zero value and
    pruned. """
    assert (splits[0], splits[-1]) == (0, csr.shape[1])
    assert sparse.isspmatrix_csr(csr)

    # make sure that the first element in every partition is explicit.
    # (this is clearly a suboptimal strategy when csr.nnz << len(splits),
    #  but it is the easiest to implement)
    new_data, new_indices, new_indptr = __sparse_row_vec__force_values_at(csr, splits[:-1])

    # now we can just call coalesce on the raw data vector
    effective_splits = np.searchsorted(new_indices, splits)
    new_data = coalesce(effective_splits, new_data, mode, fill=0)

    new_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), csr.shape)
    new_csr.eliminate_zeros()
    return new_csr

def __sparse_row_vec__force_values_at(csr, forced_indices):
    """ Given a CSR matrix, produce a ``(data, indices, indptr)`` triplet
    describing a new CSR matrix where the given indices are forced to have
    explicit values (even if they are zero).
    """
    assert sparse.isspmatrix_csr(csr)
    assert csr.shape[0] == 1

    # union of existing indices with the forced indices
    new_indices = np.insert(csr.indices, np.searchsorted(csr.indices, forced_indices), forced_indices)

    # the data at all of these indices.
    # let scipy handle this by advanced indexing the old matrix and densifying.
    new_data = np.asarray(csr[0, new_indices].todense())[0]
    new_indptr = np.array([0, len(new_indices)])

    return new_data, new_indices, new_indptr
