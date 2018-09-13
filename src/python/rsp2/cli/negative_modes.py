#!/usr/bin/env python3

# Standalone executable for the part of `rsp2` that locates negative
# modes for `SparseDiagonalizer`.

if __name__ == '__main__':
    from rsp2.internals.scipy_eigsh.negative import main_from_cli as main
    main()
else:
    raise ImportError('This script is not meant to be imported!')
