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

d = json.load(sys.stdin)
types = np.array(d.pop('types'), dtype=int)
coords = d.pop('coords')
symprec = d.pop('symprec')
assert not d
fracs = np.array(coords.pop('fracs'))
lattice = np.array(coords.pop('lattice'))

ds = spglib.get_symmetry_dataset((lattice, fracs, types), symprec)

# you can't JSON-serialize numpy types
def un_numpy_ify(d):
    if isinstance(d, dict): return {k: un_numpy_ify(v) for (k, v) in d.items()}
    if isinstance(d, list): return [un_numpy_ify(x) for x in d]
    if hasattr(d, 'tolist'): return d.tolist()
    if isinstance(d, (float, int, str)): return d
    raise TypeError

if ds:
    result = {"Ok": un_numpy_ify(ds)}
else:
    error = spglib.get_error_message()
    if error == "no error": # right...
        error = "an unspecified Bad Thing happened while calling spglib"
    result = {"Err": error}

json.dump(result, sys.stdout)
print()
