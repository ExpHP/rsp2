import json
import os
import numpy as np

def from_path(path):
    # TODO: should maybe support .tar.gz or .tar.xz
    return StructureDir.from_dir(path)

class StructureDir:
    def __init__(self, *, layers, masses, layer_sc_matrices, layer_sc_repeats, structure):
        from pymatgen.core import Structure

        # (the np.array().tolist() is to so that both arrays and lists are accepted, but lists are stored)
        self.layers = np.array(layers).tolist()
        self.masses = np.array(masses).tolist()
        self.layer_sc_matrices = np.array(layer_sc_matrices).tolist()
        self.layer_sc_repeats = np.array(layer_sc_repeats).tolist()
        assert isinstance(structure, Structure)
        self.structure = structure

    @staticmethod
    def meta_path(sdir_path):
        return os.path.join(sdir_path, 'meta.json')

    @staticmethod
    def poscar_path(sdir_path):
        return os.path.join(sdir_path, 'POSCAR')

    @classmethod
    def from_dir(cls, path):
        from pymatgen.io.vasp import Poscar

        structure = Poscar.from_file(cls.poscar_path(path)).structure
        with open(cls.meta_path(path)) as f:
            meta = json.load(f)

        layer_sc_repeats = None
        layer_sc_matrices = None
        layer_sc_matrix_dicts = meta.pop('layer_sc_matrices', None) or meta.pop('layer-sc-matrices', None)
        if layer_sc_matrix_dicts is not None:
            layer_sc_repeats = [x['repeats'] for x in layer_sc_matrix_dicts]
            layer_sc_matrices = [x['matrix'] for x in layer_sc_matrix_dicts]

        return cls(
            layers=meta.pop('layers', None),
            masses=meta.pop('masses', None),
            layer_sc_matrices=layer_sc_matrices,
            layer_sc_repeats=layer_sc_repeats,
            structure=structure,
        )

    def to_dir(self, path):
        from pymatgen.io.vasp import Poscar

        os.makedirs(path, exist_ok=True)
        Poscar(self.structure).write_file(self.poscar_path(path), significant_figures=15)

        meta = {
            'layers': self.layers,
            'masses': self.masses,
        }
        if self.layer_sc_matrices is not None:
            meta['layer-sc-matrices'] = [
                {'matrix': m, 'repeats': r}
                for (m, r) in zip(self.layer_sc_matrices, self.layer_sc_repeats)
            ]
        with open(self.meta_path(path), 'w') as f:
            json.dump(meta, f)
