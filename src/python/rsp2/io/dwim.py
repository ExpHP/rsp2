from pathlib import Path
from functools import partial
import io

__doc__ = """
Utility module that loads a variety of self-describing file formats according
to their file extension.  Automatic (de-)compression is supported through
extensions like ``.gz``.

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

def not_supported(*_args, path, **_kw):
    """
    Default callback for dispatch functions in ``to_path_impl`` and
    ``from_path_impl``.  Raises ``ValueError``.
    """
    raise ValueError(f"{repr(path)}: Unsupported file extension")

def sparse_from_npz(file, **_kw):
    """
    Possible dispatch function for ``from_path_impl``'s ``from_npz``.

    Reads a scipy sparse matrix.
    """
    import scipy.sparse
    return scipy.sparse.load_npz(file)

def sparse_to_npz(file, obj, **_kw):
    """
    Possible dispatch function for ``to_path_impl``'s ``to_npz``.

    Writes a scipy sparse matrix.
    """
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
        raise TypeError('by default, dwim .npz is only supported for sparse')

def from_path(path):
    """
    A general-purpose file-reader for things which can have multiple different
    formats (for instance, a simple dict could have been serialized to ``.json``
    or ``.yaml`` or more, and might even be compressed as ``.json.xz``...).
    Intended for use on filenames provided as CLI arguments to scripts
    (unless they are already better serviced by another module in ``rsp2.io``).

    :param path:
    A filepath to be read. Its extension will determine the behavior of
    this function.
    """
    return from_path_impl(
        path,
        from_dict=lambda x, path: x,
        from_ext={
            '.npz': sparse_from_npz,
        },
    )

def to_path(path, obj):
    """
    A general-purpose file-writer for things which can have multiple different
    formats (for instance, a simple dict can be serialized to ``.json``
    or ``.yaml`` or more, and may even be compressed as ``.json.xz``...).
    Intended for use on filenames provided as CLI arguments to scripts
    (unless they are already better serviced by another module in ``rsp2.io``).

    :param path:
    A filepath to be written.  Its extension will determine the behavior of
    this function.
    :param obj:
    The data to write.
    """
    return to_path_impl(
        path, obj,
        to_dict=lambda x, path: x,
        to_ext={
            '.npz': sparse_to_npz,
        },
    )

def from_path_impl(
        path,
        file=None,
        fullpath=None,
        from_dict=not_supported,
        from_ext=None,
):
    """
    Utility method for reading files which may have multiple formats.
    Automatically decompresses archived files and deserializes general
    serialization formats like JSON and YAML.

    :param path: a filepath from which some extensions may have been removed.
    Its extension will determine the behavior of this function.

    :param file: a filelike object in ``'rb'`` mode representing the file.
    You generally shouldn't need to provide this argument; it exists mostly as
    part of the implementation.

    :param fullpath:
    True original filename, used in error messages.
    You don't need to provide this argument.
    It will default to ``path``.

    :param from_dict:
    Function to dispatch to on ``.json`` and ``.yaml`` extensions.
    Will be called as ``from_dict(cereal, path=fullpath)``, where ``cereal``
    is the deserialized value (usually a ``dict``).

    :param from_ext:
    A dict of ``str`` to functions containing functions to dispatch to on
    unknown extensions.  The strings should include the leading dot.

    The functions will be called as ``function(file, path=fullpath)``.
    ``path`` is only provided for error messages; don't try to open it!

    :return:
    """
    from_ext = from_ext or {}

    if isinstance(path, Path):
        path = str(path)

    if fullpath is None:
        fullpath = path

    recurse = partial(from_path_impl, fullpath=fullpath, from_dict=from_dict, from_ext=from_ext)

    if file is None:
        with open(path, 'rb') as file:
            return recurse(path, file)

    if _endswith_nocase(path, '.json'):
        import json
        return from_dict(json.load(wrap_text(file)), path=fullpath)

    elif _endswith_nocase(path, '.yaml'):
        import rsp2.io._yaml_shim as yaml
        return from_dict(yaml.load(wrap_text(file)), path=fullpath)

    elif _endswith_nocase(path, '.npy'):
        import numpy
        return numpy.load(file)

    elif _endswith_nocase(path, '.gz'):
        import gzip
        return recurse(path[:-len('.gz')], gzip.GzipFile(path, fileobj=file))

    elif _endswith_nocase(path, '.xz'):
        import lzma
        return recurse(path[:-len('.xz')], lzma.LZMAFile(file))

    else:
        for ext, function in from_ext.items():
            if _endswith_nocase(path, ext):
                return function(file, path=fullpath)

        raise ValueError(f'unknown extension in {repr(fullpath)}')

def to_path_impl(
        path,
        obj,
        file=None,
        fullpath=None,
        to_dict=not_supported,
        to_ext=None,
):
    """
    Utility method for writing a file to one of multiple formats.
    Automatically performs JSON/YAML serialization, and compresses data
    based on the output extension.

    :param path: a filepath from which some extensions may have been removed.
    Its extension will determine the behavior of this function.

    :param obj: the object to write.

    :param file: a filelike object in ``'wb+'`` mode representing the file.
    If omitted, ``path`` is opened.

    You generally shouldn't need to provide this argument; it exists mostly as
    part of the implementation of extensions like `.json.xz`.

    :param fullpath:
    True original filename, used in error messages.
    You don't need to provide this argument.
    It will default to ``path``.

    :param to_dict:
    Function to dispatch to on ``.json`` and ``.yaml`` extensions.
    Will be called as ``to_dict(obj, path=fullpath)``, and should return
    a value purely composed of python primitive types.
    ``path`` is only provided for error messages; don't try to open it!

    :param to_ext:
    A dict of ``str`` to functions containing functions to dispatch to on
    unknown extensions.  The strings should include the leading dot.

    The functions will be called as ``function(file, obj, path=fullpath)``.
    ``path`` is only provided for error messages; don't try to open it!

    :return:
    """
    to_ext = to_ext or {}

    if isinstance(path, Path):
        path = str(path)

    if fullpath is None:
        fullpath = path

    recurse = partial(to_path_impl, fullpath=fullpath, to_dict=to_dict, to_ext=to_ext)

    if file is None:
        with open(path, 'wb+') as file:
            recurse(path, obj, file)
            return

    if _endswith_nocase(path, '.json'):
        import json
        if to_dict is not None:
            obj = to_dict(obj, path=fullpath)

        f = wrap_text(file)
        json.dump(obj, f)
        print(file=f) # trailing newline

    elif _endswith_nocase(path, '.npy'):
        import numpy
        numpy.save(file, obj)

    elif _endswith_nocase(path, '.gz'):
        import gzip
        recurse(path[:-len('.gz')], obj, gzip.GzipFile(path, mode='xb', fileobj=file))

    elif _endswith_nocase(path, '.xz'):
        import lzma
        recurse(path[:-len('.xz')], obj, lzma.LZMAFile(file, mode='xb'))

    else:
        for ext, function in to_ext.items():
            if _endswith_nocase(path, ext):
                function(file, obj, path=fullpath)
                return

        not_supported(path=fullpath)

def _endswith_nocase(s, suffix):
    return s[len(s) - len(suffix):].upper() == suffix.upper()

def wrap_text(f):
    if hasattr(f, 'encoding'):
        return f
    else:
        return io.TextIOWrapper(f)
