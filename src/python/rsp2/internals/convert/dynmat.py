import typing as tp

from . import _main_from_cli, DwimHooks, Pathlike, T
from rsp2.io import dynmat

def main_from_cli():
    _main_from_cli(DynmatHooks())

class DynmatHooks(DwimHooks):
    def from_path(self, path: Pathlike, *, file: tp.Optional[tp.BinaryIO] = None):
        return dynmat.from_path(path, file=file)

    def to_path(self, path: Pathlike, value: T, *, file: tp.Optional[tp.BinaryIO] = None):
        return dynmat.to_path(path, value, file=file)

    def equal(self, a: T, b: T) -> bool:
        return dynmat.equal(a, b)

if __name__ == '__main__':
    raise ImportError("This module is not an entry point!")
