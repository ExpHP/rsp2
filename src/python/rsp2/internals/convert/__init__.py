import os
import sys
import argparse
import json
import pathlib
import typing as tp

Pathlike = tp.Union[pathlib.Path, str]
T = tp.TypeVar('T')

def _main(
        from_path: tp.Callable[[Pathlike], None],
        to_path: tp.Callable[[Pathlike, T], None],
        equal: tp.Callable[[T, T], bool], # because AAAAGHGHGHG NUMPY
        keep: bool,
        input: Pathlike,
        output: Pathlike,
):
    if os.path.exists(output):
        if os.path.samefile(output, input):
            return

    m = from_path(input)
    to_path(output, m)

    # be paranoid
    if not equal(m, from_path(output)):
        info("Internal error; output does not match input", file=sys.stderr)
        sys.exit(1)

    if not keep:
        os.unlink(input)

def _main_from_cli(
        from_path: tp.Callable[[Pathlike], None],
        to_path: tp.Callable[[Pathlike, T], None],
        equal: tp.Callable[[T, T], bool],
):
    p = argparse.ArgumentParser()
    p.add_argument('INPUT')
    p.add_argument('--output', '-o', required=True)
    p.add_argument('--keep', action='store_true')
    args = p.parse_args()

    _main(
        from_path=from_path,
        to_path=to_path,
        equal=equal,
        keep=args.keep,
        output=args.output,
        input=args.INPUT,
    )

def _main_from_rust(
        from_path: tp.Callable[[Pathlike], None],
        to_path: tp.Callable[[Pathlike, T], None],
        equal: tp.Callable[[T, T], bool],
):
    d = json.load(sys.stdin)
    keep = d.pop('keep')
    output = d.pop('output')
    input = d.pop('input')
    assert not d

    _main(
        from_path=from_path,
        to_path=to_path,
        equal=equal,
        keep=keep,
        output=output,
        input=input,
    )
    print("null")
