#!/usr/bin/env python3

# Runs lammps on a poscar and generates an expected output file

import rsp2.internals.potential_test_util as util

from os.path import join
import os
import json
import tempfile
import sys
import shutil
import subprocess

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('INPUT')
    p.add_argument('-k', '--keep-temp', action='store_true')
    lmp = util.CallLammps(p)

    args = p.parse_args()
    lmp.read_arguments(args)

    tempdir = tempfile.mkdtemp()
    try:
        EXTENSION = '.pot.in'
        input = args.INPUT
        if not input.endswith(EXTENSION):
            p.error(f"Input must be a {EXTENSION} file.")
        input = input[:-len(EXTENSION)]

        _main(temp=tempdir, input_base=input, lmp=lmp)
    finally:
        if args.keep_temp:
            print(f'Kept tempdir at {tempdir}')
        else:
            shutil.rmtree(tempdir)

def _main(temp, input_base, lmp):
    populate_tempdir(temp, input_base)
    out = perform_call(temp, lmp)
    d = read_output(temp, out)
    json.dump(d, sys.stdout)
    print()

def populate_tempdir(temp, input_base):
    import lzma
    shutil.copyfile('lmp.in', join(temp, 'lmp.in'))
    shutil.copyfile(input_base + '.pot.in', join(temp, 'pot.in'))

    structure_path = os.path.join('../structure', os.path.basename(input_base))
    data = subprocess.check_output(['python3', 'generate.py', structure_path])
    with open(join(temp, 'structure.data'), 'wb') as fout:
        fout.write(data)

    with lzma.open('CC.KC.xz', 'rt') as fin:
        with open(join(temp, 'CC.KC'), 'w') as fout:
            fout.write(fin.read())

def perform_call(temp, lmp):
    env = os.environ.copy()
    env['LAMMPS_POTENTIALS'] = temp

    out = lmp.check_output_or_die(['lmp', '-i', 'lmp.in'], cwd=temp, env=env)
    print(out, file=sys.stderr)

    return out

def read_output(temp, out):
    value = util.parse_potential_from_lmp_stdout(out)
    with open(join(temp, 'dump.force')) as f:
        force = util.parse_force_dump(f)

    return {
        'value': value,
        'grad': (-force).tolist(),
    }

if __name__ == '__main__':
    main()
