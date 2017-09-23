/* Copyright (C) 2017 Michael Lamparski
 * This file is provided under the terms of the MIT License.
 */



type Lattice = ((f64, f64, f64), (f64, f64, f64));
pub(crate) fn diagonal_supercell(supercell: (i64, i64, i64), lattice: Lattice, fracs: &[f64]) -> (Lattice, Vec<f64>) {
    assert_eq!(fracs.len() % 3, 0);
    let ((xx,yy,zz), (xy,xz,yz)) = lattice;
    let (sx, sy, sz) = (supercell.0 as f64, supercell.1 as f64, supercell.2 as f64);

    let mut out = vec![];
    for ix in 0..supercell.0 {
        for iy in 0..supercell.1 {
            for iz in 0..supercell.2 {
                for i in 0..fracs.len() / 3 {
                    out.push((fracs[3*i + 0] + ix as f64) / sx);
                    out.push((fracs[3*i + 1] + iy as f64) / sy);
                    out.push((fracs[3*i + 2] + iz as f64) / sz);
                }
            }
        }
    }
    (((xx*sx, yy*sy, zz*sz), (xy*sy, xz*sz, yz*sz)), out)
}

pub(crate) fn cartesian(lattice: Lattice, fracs: &[f64]) -> Vec<f64> {
    assert_eq!(fracs.len() % 3, 0);
    let ((xx,yy,zz), (xy,xz,yz)) = lattice;

    let mut out = vec![];
    for i in 0..fracs.len() / 3 {
        let (a, b, c) = (fracs[3*i], fracs[3*i + 1], fracs[3*i + 2]);
        out.push(xz * c + xy * b + xx * a);
        out.push(yz * c + yy * b);
        out.push(zz * c);
    }
    out
}