import numpy as np
from . import dwim

def to_path(path, esols):
    """

    :param path: filepath
    :param esols: tuple of ``(eigenvalues, eigenvectors)``, where
    ``eigenvectors`` has column vectors stored as rows.
    :return:
    """
    dwim.to_path(path, to_cereal(esols))

def to_cereal(esols):
    evals, evecs = map(np.array, esols)
    if len(evals) != len(evecs):
        raise ValueError(f'length mismatch: {len(evals)} evals, {len(evecs)} evecs')
    evals = evals.tolist()
    real, imag = evecs.real.tolist(), evecs.imag.tolist()

    return evals, (real, imag)

