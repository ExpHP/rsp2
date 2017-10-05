/* Copyright (C) 2017 Michael Lamparski
 * This file is provided under the terms of the MIT License.
 */

// FIXME: file is unused

//! # A note on conventions
//!
//! There are two independent dimensions by which we describe the layout
//! of a matrix-like data structure.
//!
//! The first is the mathematical convention:
//!
//! * **column-centric** -
//!    This is the traditional convention that you probably learned in school.
//!    Most things we care about are represented by column vectors.
//!    Matrix operators are applied on the left of their "argument" (like functions).
//!    Matrix products read most naturally from right to left.
//! * **row-centric** -
//!    Most things we care about are represented by row vectors.
//!    Matrix operators are applied on the right of their "argument" (like methods),
//!    Matrix products read most naturally from left to right.
//!
//! The second is the storage convention:
//!
//! * **row-major** - The slower (outer) index is ascribed to rows.
//! * **column-major** - The slower (outer) index is ascribed to columns.
//!
//! This API uses **row-major layout _exclusively_.** The only reason we even bring
//! it up is to help clarify the distinction between row-centric and column-centric.
//!
//! Using row-major layout exclusively means that the textual rows in literal and
//! string-formatted matrices always corresponds to the rows in the mathematical object.
//! You don't need to transpose anything in your head; that would be _heartless!_
//! This decision in turn makes the row-centric formalism even more attractive,
//! because it is easy to access rows as `&[f64]` or `&[f64; 3]` in row-major layout,
//! and impossible to access columns.
//!
//! Therefore, **almost all matrices are row-centric.**  The primary exceptions are
//! *cartesian operators* and *eigenvector matrices*, each of whose column-centeredness
//! is so deeply ingrained into the mathematical community that the cost of betraying
//! convention far outweighs the benefit of easy access to each constituent vector.
mod operations {
    use ::structure_tools::array_slice::prelude::*;
    use ndarray::prelude::*;

    /// Apply a cartesian operator to positions.
    ///
    /// This essentially computes `x' = x . R^T`.
    ///
    /// Note that cartesian operators are **column-centric**, but stored **row-major**.
    /// (see the note on conventions (TODO: link)).
    pub fn apply_cartesian(carts: &[[f64; 3]], oper: &[[f64; 3]; 3]) -> Vec<[f64; 3]> {
        panic!("TODO: needs test");
        let out = aview2(carts).dot(&aview2(oper).t());
        array_to_vecs3(out.view())
    }

    /// Apply a cartesian operator to positions "in-place".
    ///
    /// Don't expect to save any memory over the non-mut version; this is moreso for
    /// conveniences such as applying a rotation to a range of positions.
    ///
    /// Note that cartesian operators are **column-centric**, but stored **row-major**.
    /// (see the note on conventions (TODO: link)).
    pub fn apply_cartesian_mut(carts: &mut [[f64; 3]], oper: &[[f64; 3]; 3]) {
        panic!("TODO: needs test");
        let temp = aview2(carts).dot(&aview2(oper).t());
        aview_mut_vecs3(carts).assign(&temp);
    }

    pub fn translate_mut(carts: &mut [[f64; 3]], alpha: f64, t: &[f64; 3]) {
        panic!("TODO: needs test");
        aview_mut_vecs3(carts).scaled_add(alpha, &aview1(t));
    }

    pub fn translate(carts: &[[f64; 3]], alpha: f64, t: &[f64; 3]) -> Vec<[f64; 3]> {
        panic!("TODO: needs test");
        let mut carts = carts.to_vec();
        translate_mut(&mut carts, alpha, t);
        carts
    }

    // TODO: replace all calls to this with 'aview_mut2' once it exists
    fn aview_mut_vecs3(carts: &mut [[f64; 3]]) -> ArrayViewMut2<f64> {
        let n = carts.len();
        aview_mut1(carts.flat_mut()).into_shape((n, 3)).unwrap()
    }

    fn array_to_vecs3(arr: ArrayView2<f64>) -> Vec<[f64; 3]> {
        assert_eq!(arr.cols(), 3);
        match arr.into_slice() {
            Some(arr) => arr.nest().to_vec(),
            None => unimplemented!(), // TODO: fallback for non-standard layout
        }
    }
}

/// Error type returned where a matrix was degenerate.
pub struct DegenerateMatrix;

/// Compute the number of unique unit cells along each direction of a supercell.
///
/// This is an integer 3-vector `[l, m, n]` such that the cartesian product
/// `(0..l) x (0..m) x (0..n)` contains exactly one image of each lattice point
/// that is distinct under translations by the supercell.
///
/// Notice that the naive method of reading the main diagonal is wholly incorrect
/// for general supercells, and in fact, the answer is not necessarily unique.
/// Be aware that future versions of this library may return a different result.
pub fn supercell_period(
    supercell: &[[i32; 3]; 3],
) -> Result<[i32; 3], DegenerateMatrix>
{
    use self::matrix_3::DeterminantExt;
    match supercell.determinant() {
        0 => Err(DegenerateMatrix),
        det => Ok({
            let out = supercell_period_unchecked(supercell);
            debug_assert_eq!(det.abs(), out[0] * out[1] * out[2]);
            out
        }),
    }
}

fn supercell_period_unchecked(
    supercell: &[[i32; 3]; 3],
) -> [i32; 3]
{
    unimplemented!()
}

pub const IDENTITY_F64: [[f64; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

fn supercell_positions_integer(
    supercell_period: &[i32; 3],
) -> Result<Vec<[i32; 3]>, DegenerateMatrix>
{
    let mut out = vec![];
    for i in 0..supercell_period[0] {
        for j in 0..supercell_period[1] {
            for k in 0..supercell_period[2] {
                out.push([i, j, k]);
            }
        }
    }
    Ok(out)
}

/// Get a set of lattice points such that every unique point
/// under a supercell is represented exactly once.
///
/// The set of points returned will not all necessarily be
/// reduced within the supercell.
pub fn supercell_positions_cart(
    supercell: &[[i32; 3]; 3],
    supercell_period: Option<&[i32; 3]>,
    lattice: &[[f64; 3]; 3],
    carts: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, DegenerateMatrix>
{
    let supercell_period = supercell_period.unwrap_or_else(|| supercell_period(supercell));
    let ints = supercell_positions_integer(&supercell_period);

    unimplemented!()
}




