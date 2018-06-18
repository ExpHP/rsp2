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
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence

import _rsp2

d = json.load(sys.stdin)
kw = d.pop('kw')
kw['allow_fewer_solutions'] = d.pop('allow-fewer-solutions')
m = _rsp2.build_input_matrix(d.pop('matrix'))
assert not d

esols = _rsp2.eigsh_custom(m, **kw)

_rsp2.emit_to_stdout(esols)
