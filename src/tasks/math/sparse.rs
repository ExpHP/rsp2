/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

//! Basic definitions of sparse matrix formats and conversions between them.
//!
//! These are currently pretty loose abstractions with public data members.
//! There is not a lot of code in rsp2 that requires sparse matrices, and seldom
//! is the same operation needed more than once.
//!
//! NOTE: These types might even be used with T = M33 or similar to represent a block
//! format, which is totally cool for a storage format but could easily lead to
//! generic mathematical operations doing the wrong thing (if written with poorly
//! chosen generic bounds). That's why they don't provide very many mathematical
//! operations.

use ::FailResult;
use ::std::collections::BTreeMap;
use ::std::ops::{Range, Add, AddAssign};
use ::num_traits::Zero;
use ::rsp2_newtype_indices::{Idx, Indexed};

pub trait VeclikeIterator: Iterator + ExactSizeIterator + DoubleEndedIterator + ::std::iter::FusedIterator {}
impl<I> VeclikeIterator for I
where I: Iterator + ExactSizeIterator + DoubleEndedIterator + ::std::iter::FusedIterator {}

//=============================================================================================

/// Coordinate format.
///
/// Elements in arbitrary order. Duplicates are allowed; they are
/// implicitly summed.  Great for building a matrix as a sum of terms,
/// but otherwise impossible to work with.
///
/// This type is "Raw" because its members are public and its invariants
/// are not protected. It exists to document the "intent" of some fields
/// of a struct, and to provide some simple utility methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawCoo<T, R: Idx = usize, C: Idx = R> {
    pub dim: (usize, usize),
    pub val: Vec<T>,
    pub row: Vec<R>,
    pub col: Vec<C>,
}

impl<T, R: Idx, C: Idx> RawCoo<T, R, C> {
    // Check properties that would typically be considered invariants of the format.
    // It is a logic error to call other methods on this type when these properties
    // are not satisfied.
    pub fn validate(&self) -> FailResult<()> {
        let RawCoo { dim, ref val, ref row, ref col } = *self;
        ensure!(row.iter().max().map(|x| x.index() < dim.0).unwrap_or(true), "row out of range");
        ensure!(col.iter().max().map(|x| x.index() < dim.1).unwrap_or(true), "col out of range");
        ensure!(val.len() == row.len(), "len mismatch: val vs row");
        ensure!(val.len() == col.len(), "len mismatch: val vs col");
        Ok(())
    }

    pub fn map<F, T2>(self, f: F) -> RawCoo<T2, R, C>
    where F: FnMut(T) -> T2,
    {
        let RawCoo { dim, val, col, row } = self;
        let val = val.into_iter().map(f).collect();
        RawCoo { dim, val, col, row }
    }

    /// Transpose R and C, disregarding T.  (i.e. if T happens to be some kind of matrix
    /// block type like M33, it will not be affected by this operation)
    pub fn into_raw_transpose(self) -> RawCoo<T, C, R> {
        let RawCoo { dim, val, row, col } = self;
        let (row, col) = (col, row);
        RawCoo { dim, val, row, col }
    }
}

impl<T, R: Idx, C: Idx> Add for RawCoo<T, R, C> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        let RawCoo { dim, val, row, col } = other;

        // addition is implied in COO format, so......
        assert_eq!(self.dim, dim);
        self.val.extend(val);
        self.row.extend(row);
        self.col.extend(col);
        self
    }
}

impl<T, R: Idx, C: Idx> RawCoo<T, R, C> {
    pub fn into_csr(self) -> RawCsr<T, R, C>
    where T: AddAssign,
    { self.into_csr_with(AddAssign::add_assign) }

    pub fn into_bee(self) -> RawBee<T, R, C>
    where T: AddAssign,
    { self.into_bee_with(AddAssign::add_assign) }

    pub fn into_indexed_dense(self) -> Indexed<R, Vec<Indexed<C, Vec<T>>>>
    where T: Zero + AddAssign,
    { self.into_indexed_dense_with(Zero::zero, AddAssign::add_assign) }

    pub fn into_dense(self) -> Vec<Vec<T>>
    where T: Zero + AddAssign,
    { self.into_dense_with(Zero::zero, AddAssign::add_assign) }

    pub fn into_csr_with<F>(self, add_assign: F) -> RawCsr<T, R, C>
    where F: FnMut(&mut T, T),
    {
        self.into_bee_with(add_assign).into_csr() // chicken
    }

    pub fn into_bee_with<F>(self, mut add_assign: F) -> RawBee<T, R, C>
    where F: FnMut(&mut T, T),
    {
        if cfg!(debug_assertions) {
            self.validate().unwrap();
        }
        let dim = self.dim;
        let mut map = BTreeMap::new();
        zip_eq!(self.val, self.row, self.col)
            .for_each(|(m, r, c)| {
                let mut m = Some(m);
                map.entry(r)
                    .or_insert_with(BTreeMap::new)
                    .entry(c)
                    .and_modify(|dest| add_assign(dest, m.take().unwrap()))
                    .or_insert_with(|| m.take().unwrap());
                assert!(m.is_none());
            });
        RawBee { dim, map }
    }

    pub fn into_indexed_dense_with<Z, F>(self, mut zero: Z, mut add_assign: F) -> Indexed<R, Vec<Indexed<C, Vec<T>>>>
    where
        Z: FnMut() -> T,
        F: FnMut(&mut T, T),
    {
        let dim = self.dim;
        let mut zero_row = || (0..dim.1).map(|_| zero()).collect();
        let mut zero_mat = || (0..dim.0).map(|_| zero_row()).collect();

        let mut out: Indexed<R, Vec<Indexed<C, Vec<T>>>> = zero_mat();
        for (r, c, x) in zip_eq!(self.row, self.col, self.val) {
            add_assign(&mut out[r][c], x);
        }
        out
    }

    pub fn into_dense_with<Z, F>(self, zero: Z, add_assign: F) -> Vec<Vec<T>>
    where
        Z: FnMut() -> T,
        F: FnMut(&mut T, T),
    {
        self.into_indexed_dense_with(zero, add_assign)
            .into_iter().map(|v| v.raw).collect()
    }

    pub fn from_dense(mat: Vec<Vec<T>>) -> Self
    where
        T: Zero,
    {
        let nrows = mat.len();
        let ncols = mat.get(0).expect("cant sparsify matrix with no rows").len();
        let dim = (nrows, ncols);

        let (mut row, mut col, mut val) = (vec![], vec![], vec![]);
        for (r, row_vec) in mat.into_iter().enumerate() {
            assert_eq!(row_vec.len(), ncols);
            for (c, x) in row_vec.into_iter().enumerate() {
                if !x.is_zero() {
                    row.push(R::new(r));
                    col.push(C::new(c));
                    val.push(x);
                }
            }
        }
        RawCoo { dim, row, col, val }
    }
}

#[allow(unused)]
impl<T, R: Idx, C: Idx> RawCoo<T, R, C>
where
    T: Clone,
{
    // these are currently just here as a placebo.  If one of them ends
    // up becoming a hot function it can be optimized to remove the clone
    // of `self`.

    pub fn to_raw_transpose(&self) -> RawCoo<T, C, R>
    { self.clone().into_raw_transpose() }

    pub fn to_csr(&self) -> RawCsr<T, R, C>
    where T: AddAssign,
    { self.clone().into_csr() }

    pub fn to_bee(&self) -> RawBee<T, R, C>
    where T: AddAssign,
    { self.clone().into_bee() }

    pub fn to_indexed_dense(&self) -> Indexed<R, Vec<Indexed<C, Vec<T>>>>
    where T: Zero + AddAssign,
    { self.clone().into_indexed_dense() }

    pub fn to_dense(&self) -> Vec<Vec<T>>
    where T: Zero + AddAssign,
    { self.clone().into_dense() }

    pub fn to_csr_with<F>(&self, add_assign: F) -> RawCsr<T, R, C>
    where F: FnMut(&mut T, T),
    { self.clone().into_csr_with(add_assign) }

    pub fn to_bee_with<F>(&self, mut add_assign: F) -> RawBee<T, R, C>
    where F: FnMut(&mut T, T),
    { self.clone().into_bee_with(add_assign) }

    pub fn to_indexed_dense_with<Z, F>(&self, mut zero: Z, mut add_assign: F) -> Indexed<R, Vec<Indexed<C, Vec<T>>>>
    where
        Z: FnMut() -> T,
        F: FnMut(&mut T, T),
    { self.clone().into_indexed_dense_with(zero, add_assign) }

    pub fn to_dense_with<Z, F>(&self, zero: Z, add_assign: F) -> Vec<Vec<T>>
    where
        Z: FnMut() -> T,
        F: FnMut(&mut T, T),
    { self.clone().into_dense_with(zero, add_assign) }
}

//=============================================================================================

/// Compressed sparse row.
///
/// Everybody's favorite.  Great for mathematical operations that must
/// traverse the entire matrix.  Poorly-equipped for in-place modifications.
///
/// This type is "Raw" because its members are public and its invariants
/// are not protected. It exists to document the "intent" of some fields
/// of a struct, and to provide some simple utility methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawCsr<T, R: Idx = usize, C: Idx = R> {
    pub dim: (usize, usize),
    pub val: Vec<T>,
    pub row_ptr: Indexed<R, Vec<usize>>,
    pub col: Vec<C>,
}

impl<T, R: Idx, C: Idx> RawCsr<T, R, C> {
    // Check properties that would typically be considered invariants of the format.
    // It is a logic error to call other methods on this type when these properties
    // are not satisfied.
    pub fn validate(&self) -> FailResult<()> {
        let RawCsr { dim, ref val, ref col, ref row_ptr } = *self;
        ensure!(val.len() == col.len(), "mismatched val/col len");
        ensure!(col.iter().max().unwrap().index() < dim.1, "col index out of range");
        ensure!(row_ptr.raw.windows(2).all(|w| w[0] <= w[1]), "row_ptr not sorted");
        ensure!(row_ptr.len() == dim.0 + 1, "row_ptr wrong len");
        ensure!(*row_ptr.raw.first().unwrap() == 0, "row_ptr first value incorrect");
        ensure!(*row_ptr.raw.last().unwrap() == val.len(), "row_ptr last value incorrect");
        ensure!(
            self.row_ranges().into_iter().all(|range| col[range].windows(2).all(|w| w[0] < w[1])),
            "columns in at least one row not strictly sorted"
        );
        Ok(())
    }

    pub fn map<F, T2>(self, f: F) -> RawCsr<T2, R, C>
    where F: FnMut(T) -> T2,
    {
        let RawCsr { dim, val, col, row_ptr } = self;
        let val = val.into_iter().map(f).collect();
        RawCsr { dim, val, col, row_ptr }
    }

    /// Transpose R and C, disregarding T.  (i.e. if T happens to be some kind of matrix
    /// block type like M33, it will not be affected by this operation)
    pub fn into_raw_transpose(self) -> RawCsr<T, C, R> {
        self.into_coo().into_raw_transpose()
            .into_csr_with(|_, _| panic!("(logic error) csr indices were not unique"))
    }

    // Vec to not extend the borrow of self
    pub fn row_ranges(&self) -> Indexed<R, Vec<Range<usize>>> {
        self.row_ptr.raw.windows(2).map(|w| w[0]..w[1]).collect()
    }
}

impl<T, R: Idx, C: Idx> RawCsr<T, R, C> {
    pub fn into_coo(mut self) -> RawCoo<T, R, C> {
        let dim = self.dim;
        let (mut row, mut col, mut val) = (vec![], vec![], vec![]);
        for (r, range) in self.row_ranges().into_iter_enumerated().rev() {
            for (x, c) in zip_eq!(self.val.drain(range.clone()), self.col.drain(range)) {
                row.push(r);
                col.push(c);
                val.push(x);
            }
        }
        assert_eq!(self.val.len(), 0);
        assert_eq!(self.col.len(), 0);
        RawCoo { dim, row, col, val }
    }
}

#[allow(unused)]
impl<T, R: Idx, C: Idx> RawCsr<T, R, C>
where
    T: Clone,
{
    // these are currently just here as a placebo.  If one of them ends
    // up becoming a hot function it can be optimized to remove the clone
    // of `self`.

    pub fn to_raw_transpose(&self) -> RawCsr<T, C, R>
    { self.clone().into_raw_transpose() }

    pub fn to_coo(&self) -> RawCoo<T, R, C>
    { self.clone().into_coo() }
}


//=============================================================================================

/// BTreeMap-based representation.
///
/// An alternative to Csr that is better at supporting in-place modification
/// (and which is highly ergonomical to iterate over), at the cost of
/// containing many more separate memory allocations.
///
/// This type is "Raw" because its members are public and its invariants
/// are not protected. It exists to document the "intent" of some fields
/// of a struct, and to provide some simple utility methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawBee<T, R: Idx = usize, C: Idx = R> {
    pub dim: (usize, usize),
    pub map: BTreeMap<R, BTreeMap<C, T>>,
}

impl<T, R: Idx, C: Idx> RawBee<T, R, C> {
    // Check properties that would typically be considered invariants of the format.
    // It is a logic error to call other methods on this type when these properties
    // are not satisfied.
    pub fn validate(&self) -> FailResult<()> {
        let RawBee { dim, ref map } = *self;
        ensure!(map.keys().max().unwrap().index() < dim.0, "row out of range");
        ensure!(map.values().filter_map(|m| m.keys().max()).max().unwrap().index() < dim.1, "col out of range");
        Ok(())
    }
}

impl<T, R: Idx, C: Idx> RawBee<T, R, C> {
    pub fn into_csr(self) -> RawCsr<T, R, C> {
        if cfg!(debug_assertions) {
            self.validate().unwrap();
        }
        let dim = self.dim;
        let mut val = vec![];
        let mut col = vec![];
        let mut row_ptr = Indexed::new();
        row_ptr.push(0);

        for (row, row_map) in self.map {
            // fill in spots for rows that aren't even in the map
            row_ptr.resize(row.index() + 1, val.len());

            for (this_col, this_val) in row_map {
                val.push(this_val);
                col.push(this_col);
            }
        }
        // potentially even more missing rows at the end
        row_ptr.resize(dim.0 + 1, val.len());

        let out = RawCsr { dim, val, row_ptr, col };
        if cfg!(debug_assertions) {
            out.validate().unwrap();
        }
        out
    }

    pub fn into_coo(self) -> RawCoo<T, R, C> {
        if cfg!(debug_assertions) {
            self.validate().unwrap();
        }
        let dim = self.dim;
        let mut val = vec![];
        let mut row = vec![];
        let mut col = vec![];
        for (r, row_map) in self.map {
            for (c, x) in row_map {
                row.push(r);
                col.push(c);
                val.push(x);
            }
        }
        RawCoo { dim, val, row, col }
    }
}

#[allow(unused)]
impl<T, R: Idx, C: Idx> RawBee<T, R, C>
where
    T: Clone,
{
    // these are currently just here as a placebo.  If one of them ends
    // up becoming a hot function it can be optimized to remove the clone
    // of `self`.

    pub fn to_csr(&self) -> RawCsr<T, R, C>
    { self.clone().into_csr() }

    pub fn to_coo(&self) -> RawCoo<T, R, C>
    { self.clone().into_coo() }
}

//=============================================================================================

#[test]
fn bee_to_csr() {
    // Test case with variety of hostile qualities:
    // * Rows that are outright not present in the map
    // * A row that is present but has no nonzero entries
    // * Rows at the *end* that are not present in the map
    use std::iter::FromIterator;

    let dim = (7, 4);
    let map = BTreeMap::from_iter(vec![
        (2, BTreeMap::from_iter(vec![
            (2, 1.0f64),
            (3, 3.0f64),
        ])),
        (3, BTreeMap::from_iter(vec![])),
        (5, BTreeMap::from_iter(vec![
            (2, 2.0f64),
        ])),
    ]);
    let bee = RawBee { dim, map };
    let RawCsr { dim, val, row_ptr, col } = bee.to_csr();
    assert_eq!(dim, (7, 4));
    assert_eq!(val, vec![1.0, 3.0, 2.0]);
    assert_eq!(col, vec![2, 3, 2]);
    assert_eq!(row_ptr.raw, vec![0, 0, 0, 2, 2, 2, 3, 3]);
}
