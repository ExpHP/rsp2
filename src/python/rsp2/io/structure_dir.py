import json
import os

def from_path(path):
    # TODO: should maybe support .tar.gz or .tar.xz
    return StructureDir.from_dir(path)

class StructureDir:
    def __init__(self, *, layers, masses, layer_sc_matrices, structure):
        self.layers = layers
        self.masses = masses
        self.layer_sc_matrices = layer_sc_matrices
        self.structure = structure

    @classmethod
    def from_dir(cls, path):
        from pymatgen.io.vasp import Poscar

        structure = Poscar.from_file(os.path.join(path, 'POSCAR')).structure
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)

        layer_sc_matrices = meta.pop('layer_sc_matrices', None) or meta.pop('layer-sc-matrices', None)
        if layer_sc_matrices:
            layer_sc_matrices = [x['matrix'] for x in layer_sc_matrices]

        return cls(
            layers=meta.pop('layers', None),
            masses=meta.pop('masses', None),
            layer_sc_matrices=layer_sc_matrices,
            structure=structure,
        )
