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

from . import eigsh_custom
from rsp2.io import dynmat
from rsp2.io import eigensols

def main(d):
    kw = d.pop('kw')
    kw['allow_fewer_solutions'] = d.pop('allow-fewer-solutions')
    m = dynmat.from_dict(d.pop('matrix'))
    assert not d

    return eigsh_custom(m, **kw)

if __name__ == '__main__':
    json.dump(eigensols.to_cereal(main(json.load(sys.stdin))), sys.stdout)
    print(file=sys.stdout)
