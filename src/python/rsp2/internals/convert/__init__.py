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

    if not keep:
        os.unlink(input)

def _main_from_cli(
        hooks: DwimHooks,
):
    p = argparse.ArgumentParser()

    p.add_argument('INPUT')
    p.add_argument('--output', '-o', required=True)
    p.add_argument('--stdin', action='store_true', help='read from stdin, using the file extension of INPUT to determine the format')

    p.add_argument('--stdout', action='store_true', help='write to stdout, using the file extension of --output to determine the format')
    p.add_argument('--keep', action='store_true')
    args = p.parse_args()

    _main(
        hooks=hooks,
        keep=args.keep,
        output=args.output,
        input=args.INPUT,
        input_is_stdin=args.stdin,
        output_is_stdout=args.stdout,
    )

def _main_from_rust(
        hooks: DwimHooks,
):
    d = json.load(sys.stdin)
    keep = d.pop('keep')
    output = d.pop('output')
    input = d.pop('input')
    assert not d

    _main(
        hooks=hooks,
        keep=keep,
        output=output,
        input=input,
        input_is_stdin=False,
        output_is_stdout=False,
    )
    print("null")
