#!/usr/bin/env python3

###########################################################################
#  This file is part of rsp2, and is licensed under EITHER the MIT license
#  or the Apache 2.0 license, at your option.
#
#      http://www.apache.org/licenses/LICENSE-2.0
#      http://opensource.org/licenses/MIT
#
#  Be aware that not all of rsp2 is provided under this permissive license,
#  and that the project as a whole is licensed under the GPL 3.0.
###########################################################################

import json
import sys
import numpy as np
import spglib

def main():
    d = json.load(sys.stdin)
    result = _result_main(**d)
    json.dump(result, sys.stdout)
    print()

def _result_main(**kw):
    try:
        return _except_main(**kw)
    except SpgError as e:
        return {"Err": "spglib: " + e.message() }

def _except_main(types, coords, symprec):
    types = np.array(types, dtype=int)
    fracs = np.array(coords.pop('fracs'))
    lattice = np.array(coords.pop('lattice'))

    cell = (lattice, fracs, types)

    # check primitiveness
    # alas, SPGLIB's symmetry dataset does not give any reliable way to
    # check this.
    #
    # FIXME this should be checked sooner in RSP2, not after expensive relaxation
    prim = spglib.find_primitive(cell, symprec=symprec)
    if not prim:
        SpgError.throw()

    prim_lattice = prim[0]
    vol_ratio = abs(np.linalg.det(lattice) / np.linalg.det(prim_lattice))
    if abs(abs(vol_ratio) - 1) > 1e-4:
        return {"Err": "rsp2 requires the input to be a primitive cell"}

    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if not ds:
        SpgError.throw()

    # you can't JSON-serialize numpy types
    def un_numpy_ify(d):
        if isinstance(d, dict): return {k: un_numpy_ify(v) for (k, v) in d.items()}
        if isinstance(d, list): return [un_numpy_ify(x) for x in d]
        if hasattr(d, 'tolist'): return d.tolist()
        if isinstance(d, (float, int, str)): return d
        raise TypeError

    return {"Ok": un_numpy_ify(ds)}

class SpgError(RuntimeError):
    @classmethod
    def throw(cls):
        error = spglib.get_error_message()
        if error == "no error": # right...
            error = "an unspecified Bad Thing happened while calling spglib"
        raise cls(error)

    def message(self):
        return self.args[0]

if __name__ == '__main__':
    main()
