from pathlib import Path
import io

__doc__ = """
Utility module that loads a variety of self-describing file formats according
to their file extension.  Automatic (de-)compression is supported through
extensions like `.gz`.

The basic idea here is that there aren't often many knobs you need to turn
to be able to read files in self-describing formats... but you do constantly
need to look up the right commands and libraries needed to read them, which is
a pain.  Furthermore, some parts of rsp2 choose rather arbitrarily between
formats like JSON and YAML based on considerations such as "where will this
file be used?" and etc, which often change over time.

This module was originally written simply to make life easier on the Python
REPL, but is also now used by RSP2 itself to let some CLI commands support a
variety of inputs.
"""

def from_path(path, file=None, fullpath=None):
    if isinstance(path, Path):
        path = str(path)

    if fullpath is None:
        fullpath = path
    if file is None:
        with open(path, 'rb') as file:
            return from_path(path, file)

    if _endswith_nocase(path, '.json'):
        import json
        return json.load(wrap_text(file))

    elif _endswith_nocase(path, '.yaml'):
        import common.rsp2.io._yaml_shim as yaml
        return yaml.load(wrap_text(file))

    elif _endswith_nocase(path, '.npy'):
        import numpy
        return numpy.load(file)

    elif _endswith_nocase(path, '.npz'):
        import scipy.sparse
        return scipy.sparse.load_npz(file)

    elif _endswith_nocase(path, '.gz'):
        import gzip
        return from_path(path[:-len('.gz')], gzip.GzipFile(path, fileobj=file))

    else:
        raise ValueError(f'unknown extension in {repr(fullpath)}')

def path_is_dict_filetype(path):
    return any(f() for f in [
        lambda: _endswith_nocase(path, '.json'),
        lambda: _endswith_nocase(path, '.yaml'),
        lambda: _endswith_nocase(path, '.yaml'),
    ])

def to_path(path, obj, file=None, fullpath=None, to_dict=None):
    if isinstance(path, Path):
        path = str(path)

    if fullpath is None:
        fullpath = path
    if file is None:
        with open(path, 'wb+') as file:
            return to_path(path, obj, file, fullpath, to_dict)

    if _endswith_nocase(path, '.json'):
        import json
        if to_dict is not None:
            obj = to_dict(obj)
        json.dump(obj, wrap_text(file))

    elif _endswith_nocase(path, '.npy'):
        import numpy
        numpy.save(file, obj)

    elif _endswith_nocase(path, '.npz'):
        import scipy.sparse
        from io import BytesIO
        if scipy.sparse.issparse(obj):
            # FIXME: This runs into not-nice errors if you try to use `.npz.gz`
            # as a format.  Basically, even if you say `compressed=False`,
            # `save_npz` goes through the zipfile interface, which attempts
            # to seek, which is not supported by the GZip IO wrapper)
            buf = BytesIO()
            scipy.sparse.save_npz(buf, obj, compressed=True)
            file.write(buf.getvalue())
        else:
            raise TypeError('dwim .npz is only supported for sparse')

    elif _endswith_nocase(path, '.gz'):
        import gzip
        to_path(path[:-len('.gz')], obj, gzip.GzipFile(path, mode='xb', fileobj=file), to_dict)

    else:
        raise ValueError(f'unknown extension in {repr(fullpath)}')

def _endswith_nocase(s, suffix):
    return s[len(s) - len(suffix):].upper() == suffix.upper()

def wrap_text(f):
    if hasattr(f, 'encoding'):
        return f
    else:
        return io.TextIOWrapper(f)
