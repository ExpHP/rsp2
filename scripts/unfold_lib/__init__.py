import subprocess
import os
import numpy as np

ROOT = os.path.dirname(__file__)
LIBDIR = os.path.join(ROOT, '../../target/release')

__all__ = [
    "BuildError",
    "build",
    "unfold_all",
]

unfold_all = None
diagonalize_dynmat = None

class BuildError(Exception):
    """ Indicates that the rust extension cannot be built or linked for some reason. """
    pass

def build():
    global unfold_all

    import ctypes

    try:
        subprocess.check_call(['cargo', 'build', '--release'], cwd=ROOT)
    except (FileNotFoundError, IOError, OSError, subprocess.SubprocessError) as e:
        raise BuildError(e)

    library_path = os.path.join(LIBDIR, 'librsp2c_unfold.so')
    try:
        imp = ctypes.cdll.LoadLibrary(library_path)
    except Exception as e: # I don't know what this can throw on linker errors
        if isinstance(e, (TypeError, NameError, AttributeError, ValueError)):
            raise e # legitimate python errors; don't wrap them
        raise BuildError(e)

    # This definition gets written to the exported global name
    def _unfold_all(
            superstructure,
            translation_carts,
            gpoint_sfracs,
            kpoint_sfrac,
            eigenvectors,
            translation_deperms,
            translation_phases,
            progress_prefix,
    ):
        from ctypes import c_double, c_int32

        super_lattice = superstructure.lattice.matrix
        super_carts = superstructure.cart_coords

        nquotient = len(translation_carts)
        nsite = len(super_carts)
        nev = len(eigenvectors)

        output = np.zeros((nev, nquotient), dtype=c_double)
        arrays = [
            # array, check_shape, [ffi_pointer_type, py_type (if different)]
            (super_lattice, (3, 3), [c_double, None]),
            (super_carts, (nsite, 3), [c_double, None]),
            (translation_carts, (nquotient, 3), [c_double, None]),
            (gpoint_sfracs, (nquotient, 3), [c_double, None]),
            (kpoint_sfrac, (3,), [c_double, None]),
            (eigenvectors, (nev, nsite, 3), [c_double, np.complex128]),
            (translation_deperms, (nquotient, nsite), [c_int32, None]),
            (translation_phases, (nquotient, nsite), [c_double, np.complex128]),
            (output, (nev, nquotient), [c_double, None]),
        ]

        pointers = []
        for (key, (array, shape, [ctype, py_type])) in enumerate(arrays):
            if py_type is None:
                py_type = ctype
            array = np.reshape(array, shape).astype(py_type, copy=False)
            array = np.ascontiguousarray(array)
            pointers.append(array.ctypes.data_as(ctypes.POINTER(ctype)))

        if progress_prefix is None:
            progress_prefix = ctypes.c_char_p(None) # Null pointer
        else:
            progress_prefix = ctypes.c_char_p(progress_prefix.encode('utf-8'))

        code = imp.rsp2c_unfold_all(
            ctypes.c_int64(nquotient),
            ctypes.c_int64(nsite),
            ctypes.c_int64(nev),
            progress_prefix,
            *pointers,
        )
        if code:
            raise RuntimeError('rsp2c failed')

        return output
    unfold_all = _unfold_all
