import subprocess
import os
import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(__file__)
LIBDIR = os.path.join(ROOT, '../../target/release')

__all__ = [
    "build",
    "unfold_all_gamma",
]

unfold_all_gamma = None

def build():
    global unfold_all_gamma

    import ctypes

    subprocess.check_call(['cargo', 'build', '--release'], cwd=ROOT)

    imp = ctypes.cdll.LoadLibrary(os.path.join(LIBDIR, 'librsp2c_unfold.so'))

    def unfold_all_gamma(
            superstructure,
            translation_carts,
            gpoint_sfracs,
            eigenvectors,
            translation_deperms,
    ):
        from ctypes import c_double, c_int32

        super_lattice = superstructure.lattice.matrix
        super_carts = superstructure.cart_coords

        nquotient = len(translation_carts)
        nsite = len(super_carts)
        nev = len(eigenvectors)
        arrays = [
            (super_lattice, (3, 3), c_double),
            (super_carts, (nsite, 3), c_double),
            (translation_carts, (nquotient, 3), c_double),
            (gpoint_sfracs, (nquotient, 3), c_double),
            (eigenvectors, (nev, nsite, 3), c_double),
            (translation_deperms, (nquotient, nsite), c_int32),
        ]

        for (key, (array, shape, ctype)) in enumerate(arrays):
            array = array.reshape(shape).astype(ctype, copy=False)
            array = np.ascontiguousarray(array)
            arrays[key] = array

        vec = imp.rsp2c_vec_new()
        imp.unfold_all_gamma(
            ctypes.c_int32(nquotient),
            ctypes.c_int32(nsite),
            ctypes.c_int32(nev),
            *arrays,
            vec,
        )

        n = imp.rsp2c_vec_len(vec)
        ptr = imp.rsp2c_vec_data(vec)
        data = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
        array = np.ctypeslib.as_array(data, (n,)).copy()

        imp.rsp2c_vec_free(vec)

        return array.reshape((nev, nquotient))
