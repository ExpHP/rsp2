import h5py
import os

def gamma_from_band_hdf5(path='band.hdf5'):
    assert os.path.exists(path)
    h = h5py.File(path)
    q = h['path'].value
    freq = h['frequency'].value
    evs = h['eigenvector'].value
    for stuff in zip(q, freq, evs):
        for (q, freq, evs) in zip(*stuff):
            if (q == 0.0).all():
                evs = evs.T
                n = len(evs) // 3
                return freq, evs.reshape(n * 3, n, 3)
    raise RuntimeError('gamma point not found in hdf5 file')

