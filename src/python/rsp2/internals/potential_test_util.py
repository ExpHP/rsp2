import numpy as np

# Requires the following:
#
# thermo_style custom pe
# thermo_modify line multi format float %.16g
def parse_potential_from_lmp_stdout(stdout):
    pe_line, = [line for line in stdout.splitlines() if 'PotEng' in line]
    assert len(pe_line.split()) == 3
    return float(pe_line.split()[2])

# dump format must be 'id fx fy fz'
#
# e.g.
# dump F all custom 1 dump.force id fx fy fz
# dump_modify F format line "%d %.16g %.16g %.16g"
def parse_force_dump(file):
    output_force = []
    lines = iter(file)
    for line in lines:
        if 'ITEM: NUMBER OF ATOMS' in line:
            break

    parts = next(lines).split()
    assert len(parts) == 1
    n_atoms = int(parts[0])

    for line in lines:
        if 'ITEM: ATOMS id fx fy fz' in line:
            break
    else:
        assert False, 'could not find beginning of data (was the dump format correct?)'

    for expected_n, line in enumerate(take(n_atoms, lines), start=1):
        parts = line.split()
        assert len(parts) == 4
        assert expected_n == int(parts[0])
        output_force.append(tuple(map(float, parts[1:1+3])))
    return np.array(output_force)

# Factors out a bit of shared logic in the `make-expected` scripts for RSP2's
# potential tests which is too terrible to deserve to appear more than once.
#
# Basically, it helps cope with "missing" error messages from lammps.
#
# NOTE: I tried subprocess's `bufsize=1` and `bufsize=0` options, and
#       unfortunately the issue persisted, so this atrocity is still necessary.
class CallLammps:
    """
    Correct usage:

    >>> import argparse, sys
    >>> parser = argparse.ArgumentParser()
    >>> lmp = CallLammps(parser) # add the --tty-hack argument
    >>>
    >>> args = parser.parse_args()
    >>> lmp.read_arguments(args) # read the argument
    >>>
    >>> # This takes keyword args like `subprocess.check_output`.
    >>> #
    >>> # When it runs, it will optionally add a '-screen' argument
    >>> # based on whether the --tty-hack flag was provided.
    >>> lmp.check_output_or_die(['lmp', '-in', 'blah.in'])
    """

    def __init__(self, parser):
        parser.add_argument(
            '--tty-hack',
            action='store_true',
            help=
                'summon ancient demonic powers to tell lammps to write directly to the current tty device (your terminal), '
                'bypassing all forms of STDOUT redirection. This may help you see error messages printed by lammps '
                'in cases where lammps forgets to flush before aborting...',
        )
        self.tty_hack = None

    def read_arguments(self, args):
        self.tty_hack = args.tty_hack

    def check_output_or_die(self, cmd, *args, **kw):
        """
        Replacement for ``subprocess.check_output(cmd, *args, **kw).decode('utf-8')``.
        """
        import subprocess, sys
        if self.tty_hack is None:
            raise TypeError("Forgot to call 'read_arguments'!")

        try:
            # LAMMPS unfortunately does not seem to flush either stdout or
            # its logfile before calling MPI_Abort, causing a great deal of
            # its stdout (including the error message!) to be lost.
            if self.tty_hack:
                tty = subprocess.check_output('tty').decode('utf-8').rstrip('\r\n')
                cmd += ['-screen', tty]

            return subprocess.check_output(cmd, *args, **kw).decode('utf-8')

        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'), file=sys.stderr)
            if self.tty_hack:
                print("lammps failed!", file=sys.stderr)
                sys.exit(1)
            else:
                print("lammps failed! (if you don't see the actual error message in the above output, try again with --tty-hack)" , file=sys.stderr)
                sys.exit(1)

def take(n, it):
    for _ in range(n):
        yield next(it)
