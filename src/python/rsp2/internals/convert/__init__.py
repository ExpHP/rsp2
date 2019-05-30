import os
import sys
import argparse
import json
import pathlib
import typing as tp
import abc
from rsp2.internals import info

Pathlike = tp.Union[pathlib.Path, str]
T = tp.TypeVar('T')

class DwimHooks(abc.ABC):
    @abc.abstractmethod
    def from_path(self, path: Pathlike, *, file: tp.Optional[tp.BinaryIO] = None) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def to_path(self, path: Pathlike, value: T, *, file: tp.Optional[tp.BinaryIO] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def equal(self, a: T, b: T) -> bool: # because AAAAGHGHGHG NUMPY
        raise NotImplementedError

def _main(
        hooks: DwimHooks,
        keep: bool,
        input: Pathlike,
        output: Pathlike,
        input_is_stdin: bool,
        output_is_stdout: bool,
):
    keep = keep or output_is_stdout

    if os.path.exists(output):
        if os.path.samefile(output, input):
            return

    if input_is_stdin:
        m = hooks.from_path(input, file=sys.stdin.buffer)
    else:
        m = hooks.from_path(input)

    if output_is_stdout:
        hooks.to_path(output, m, file=sys.stdout.buffer)
    else:
        hooks.to_path(output, m)

    # be paranoid
    if not output_is_stdout and not hooks.equal(m, hooks.from_path(output)):
        info("Internal error; output does not match input")
        sys.exit(1)

    if not input_is_stdin and not keep:
        os.unlink(input)

def _main_from_cli(
        hooks: DwimHooks,
):
    p = argparse.ArgumentParser()

    p.add_argument('INPUT')
    p.add_argument('--output', '-o', required=True)
    p.add_argument('--stdin', action='store_true', help='read from stdin, using the file extension of INPUT to determine the format. Implies --keep.')
    p.add_argument('--stdout', action='store_true', help='write to stdout, using the file extension of --output to determine the format')
    p.add_argument('--keep', action='store_true')
    args = p.parse_args()

    _main(
        hooks=hooks,
        keep=args.keep,
        input=args.INPUT,
        output=args.output,
        input_is_stdin=args.stdin,
        output_is_stdout=args.stdout,
    )

def _read_from_rust(
        hooks: DwimHooks,
):
    p = argparse.ArgumentParser()
    p.add_argument('INPUT')
    args = p.parse_args()

    # Receive a unit type
    assert json.load(sys.stdin) in [None, []]

    # Send the matrix
    _main(
        hooks=hooks,
        keep=False,
        input=args.INPUT,
        output="<stdout>.json",
        input_is_stdin=False,
        output_is_stdout=True,
    )

def _write_from_rust(
        hooks: DwimHooks,
):
    p = argparse.ArgumentParser()
    p.add_argument('OUTPUT')
    args = p.parse_args()

    # Receive the matrix and write it
    _main(
        hooks=hooks,
        keep=False,
        input="<stdin>.json",
        output=args.OUTPUT,
        input_is_stdin=True,
        output_is_stdout=False,
    )
    # Send a unit type
    print("null")
