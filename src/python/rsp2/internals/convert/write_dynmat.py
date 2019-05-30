from . import _write_from_rust
from .dynmat import DynmatHooks

if __name__ == '__main__':
    _write_from_rust(DynmatHooks())
else:
    raise ImportError("This module is only for use as an entry point!")
