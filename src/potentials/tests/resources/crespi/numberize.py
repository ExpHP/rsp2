#!/usr/bin/env python3

# Because for some bizarre reason LAMMPS requires atom indices in the input file.

import sys

def main():
    import argparse
    p = argparse.ArgumentParser(
        description=
            "Replaces '!id' appearing in consecutive lines of text with '1', "
            "'2', '3', etc. Separate runs of lines (e.g. with at least one "
            "unsubstituted line in-between) restart the numbering."
    )
    _ = p.parse_args()

    if sys.stdin.isatty() and sys.stderr.isatty():
        print("Reading from STDIN...", file=sys.stderr)

    counter = 1
    for line in sys.stdin:
        if "!id" in line:
            line = line.replace("!id", str(counter))
            counter += 1
        else:
            counter = 1

        print(line, end='')

if __name__ == '__main__':
    main()
