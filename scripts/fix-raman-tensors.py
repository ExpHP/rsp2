#!/usr/bin/env python3

import argparse
import os
import sys
import json
import numpy as np

PROG = os.path.basename(sys.argv[0])

def main():
    parser = argparse.ArgumentParser(
        description='Fix a raman.json file created before 99e7a42a5 (June 14)',
    )
    parser.add_argument('INPUT', nargs='*')
    parser.add_argument(
        '--temperature', type=float,
        help="specify temperature, for double checking that the missing prefactor matches expectation.",
    )
    parser.add_argument(
        '--mistake-no', type=int, default=1,
        help="which mistake defines our expectation?"
             " 1: first mistake (completely missing), "
             " 2: second mistake (missing sqrt)",
    )
    args = parser.parse_args()

    for path in args.INPUT:
        corrected_loc = path + '.corrected'

        with open(path) as f:
            d = json.load(f)

        frequencies = np.array(d['frequency'])
        correct_averages = np.array(d['average-3d'])
        recorded_tensors = np.array(d['raman-tensor'])
        actual_averages = np.sum(recorded_tensors**2, axis=(1,2)) / 9

        if np.allclose(correct_averages, actual_averages, rtol=1e-10, atol=0):
            continue

        missing_prefactors = correct_averages / actual_averages
        missing_prefactors[frequencies <= 0] = 0

        if args.temperature is not None:
            expected_prefactors = get_expected_prefactors(frequencies, temperature=args.temperature, mistake=args.mistake_no)
            if np.allclose(expected_prefactors, missing_prefactors, atol=0):
                warn(f"{path} has missing prefactors that match expectation")
            else:
                warn(f"{path} has missing prefactors that DO NOT match expectation!!")
                # print(np.hstack([correct_averages[:, None], actual_averages[:, None]]), file=sys.stderr)
                # print(np.hstack([expected_prefactors[:, None], missing_prefactors[:, None]]), file=sys.stderr)

        else:
            warn(f"{path} has missing prefactors")

        warn(f"Writing {corrected_loc}")

        correct_tensors = recorded_tensors * np.sqrt(missing_prefactors[:, None, None])
        d['raman-tensor'] = correct_tensors.tolist()
        with open(corrected_loc, 'w') as f:
            json.dump(d, f)
            print(file=f)

def get_expected_prefactors(frequencies, temperature, mistake):
    hk = 0.22898852319

    if temperature == 0:
        bose_occupation = 1
    else:
        expm1 = np.expm1(hk * frequencies / temperature)
        bose_occupation = (1.0 + 1.0 / expm1)
    prefactors = bose_occupation / frequencies

    if mistake == 1:
        return np.where(frequencies <= 0.0, 0.0, prefactors)
    elif mistake == 2:
        return np.where(frequencies <= 0.0, 0.0, prefactors ** -1)
    else:
        raise ValueError('mistake')

# ------------------------------------------------------

def warn(*args, **kw):
    print(f'{PROG}:', *args, file=sys.stderr, **kw)

def die(*args, code=1):
    warn('Fatal:', *args)
    sys.exit(code)

# ------------------------------------------------------

if __name__ == '__main__':
    main()
