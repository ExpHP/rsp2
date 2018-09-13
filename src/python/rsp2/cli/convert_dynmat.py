#!/usr/bin/env python3

if __name__ == '__main__':
    from rsp2.internals.convert.dynmat import main_from_cli as main
    main()
else:
    raise ImportError('This script is not meant to be imported!')
