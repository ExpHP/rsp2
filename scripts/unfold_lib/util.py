import scipy.sparse as sparse
import numpy as np
import typing as tp

A = tp.TypeVar('A')
B = tp.TypeVar('B')

def check_arrays(**kw):
    previous_values = {}

    kw = {name: list(data) for name, data in kw.items()}
    for name in kw:
        if not sparse.issparse(kw[name][0]):
            kw[name][0] = np.array(kw[name][0])

    for name, data in kw.items():
        if len(data) == 2:
            array, dims = data
            dtype = None
        elif len(data) == 3:
            array, dims, dtype = data
        else:
            raise TypeError(f'{name}: Expected (array, shape) or (array, shape, dtype)')

        if dtype:
            # support one dtype or a list of them
            if isinstance(dtype, type):
                dtype = [dtype]
            if not any(issubclass(np.dtype(array.dtype).type, d) for d in dtype):
                raise TypeError(f'{name}: Expected one of {dtype}, got {array.dtype}')

        # format names without quotes
        nice_expected = '[' + ', '.join(map(str, dims)) + ']'
        if len(dims) != array.ndim:
            raise TypeError(f'{name}: Wrong number of dimensions (expected shape {nice_expected}, got {list(array.shape)})')

        for axis, dim in enumerate(dims):
            if isinstance(dim, int):
                if array.shape[axis] != dim:
                    raise TypeError(f'{name}: Mismatched dimension (expected shape {nice_expected}, got {list(array.shape)})')
            elif isinstance(dim, str):
                if dim not in previous_values:
                    previous_values[dim] = (array.shape[axis], name, axis)

                if previous_values[dim][0] != array.shape[axis]:
                    prev_value, prev_name, prev_axis = previous_values[dim]
                    raise TypeError(
                        f'Conflicting values for dimension {repr(dim)}:\n'
                        f' {prev_name}: {kw[prev_name][0].shape} (axis {prev_axis}: {prev_value})\n'
                        f' {name}: {array.shape} (axis {axis}: {array.shape[axis]})'
                    )

    return {dim:tup[0] for (dim, tup) in previous_values.items()}

def map_with_progress(
        xs: tp.Iterator[A],
        progress: tp.Callable[[int, int], None],
        function: tp.Callable[[A], B],
) -> tp.Iterator[B]:
    yield from (function(x) for x in iter_with_progress(xs, progress))

def iter_with_progress(
        xs: tp.Iterator[A],
        progress: tp.Callable[[int, int], None],
) -> tp.Iterator[A]:
    xs = list(xs)

    for (num_done, x) in enumerate(xs):
        if progress:
            progress(num_done, len(xs))

        yield x

    if progress:
        progress(len(xs), len(xs))

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
