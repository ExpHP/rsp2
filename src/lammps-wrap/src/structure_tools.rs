/* Copyright (C) 2017 Michael Lamparski
 * This file is provided under the terms of the MIT License.
 */

// This is the module where all of your hopes and dreams die.
// You should stop looking at it.

// really dumb representation of a lammps lattice, I don't even
// know why this is here.
type Lattice = ((f64, f64, f64), (f64, f64, f64));

pub fn decode_lattice(((xx,yy,zz), (xy,xz,yz)): Lattice) -> [[f64; 3]; 3]
{
    [
        [ xx, 0.0, 0.0],
        [ xy,  yy, 0.0],
        [ xz,  yz,  zz],
    ]
}

pub fn encode_lattice(matrix: [[f64; 3]; 3]) -> Lattice
{
    assert_eq!(matrix[0][1], 0f64);
    assert_eq!(matrix[0][2], 0f64);
    assert_eq!(matrix[1][2], 0f64);
    (
        (matrix[0][0], matrix[1][1], matrix[2][2]),
        (matrix[1][0], matrix[2][0], matrix[2][1]),
    )
}

pub fn supercell_diagonal(supercell: (i64, i64, i64), lattice: Lattice, fracs: &[f64]) -> (Lattice, Vec<f64>) {
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

pub fn cartesian(lattice: Lattice, fracs: &[f64]) -> Vec<f64> {
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

pub fn fractional(lattice: Lattice, carts: &[f64]) -> Vec<f64> {
    cartesian(invert_lammps_lattice(lattice), carts)
}

fn invert_lammps_lattice(((a,b,c), (d,e,f)): Lattice) -> Lattice {
    // (just asked wolfram alpha for the inverse of a lower triangular 3x3 matrix)
    (
        (a.recip(), b.recip(), c.recip()),
        (-d / (a * b), (d*f - b*e)/(a*b*c), -f / (b * c))
    )
}
