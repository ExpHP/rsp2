import numpy as np
from . import dwim

def to_path(path, esols):
    """
    :param path: filepath
    :param esols: tuple of ``(eigenvalues, eigenvectors)``, where
    ``eigenvectors`` has column vectors stored as rows.
    :return:
    """
    return dwim.to_path_impl(
        path, esols,
        to_dict=to_cereal,
        to_ext={'.npz': to_npz},
    )

def from_path(path):
    """
    :param path: filepath
    :return:
    """
    return dwim.from_path_impl(
        path,
        from_dict=from_cereal,
        from_ext={'.npz': from_npz},
    )

# Text (JSON) format used by RSP2's rust parts,
# generally only for partial diagonalization.
def to_cereal(esols, **_kw):
    evals, evecs = map(np.array, esols)
    if len(evals) != len(evecs):
        raise ValueError(f'length mismatch: {len(evals)} evals, {len(evecs)} evecs')
    evals = evals.tolist()
    real, imag = evecs.real.tolist(), evecs.imag.tolist()

    return evals, (real, imag)

def from_cereal(cereal, **_kw):
    evals, (real, imag) = cereal
    if len(evals) != len(real):
        raise ValueError(f'length mismatch: {len(evals)} evals, {len(real)} real parts')

    return np.array(evals), np.array(real) + 1j * np.array(imag)

# Binary format used by Python code, particularly when dealing with full diagonalization
def from_npz(file, **_kw):
    with np.load(file) as npz:
        return npz.f.eigenvalues, npz.f.eigenvectors

def to_npz(file, esols, **_kw):
    evals, evecs = esols
    np.savez_compressed(file, eigenvalues=evals, eigenvectors=evecs)

def wavenumber_from_eigenvalue(eigenvalue):
    """
    Converts eigenvalue in rsp2's natural units to wavenumber in cm^-1.
    Supports numpy arrays.

    Negative eigenvalues will be represented as negative frequencies
    rather than imaginary.
    """
    eigenvalue = np.array(eigenvalue)

    # = sqrt(eV/amu)/angstrom/(2*pi)/THz
    SQRT_EIGENVALUE_TO_THZ = 15.6333043006705
    # = THz / (c / cm)
    THZ_TO_WAVENUMBER = 33.3564095198152

    return (
            SQRT_EIGENVALUE_TO_THZ * THZ_TO_WAVENUMBER
            * np.absolute(eigenvalue) ** 0.5
            * np.sign(eigenvalue)
    )
