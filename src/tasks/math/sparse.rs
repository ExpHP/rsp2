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
use ::std::ops::{Range, AddAssign};
use ::num_traits::Zero;
use ::rsp2_newtype_indices::{Idx, Indexed};

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

pub trait VeclikeIterator: Iterator + ExactSizeIterator + DoubleEndedIterator + ::std::iter::FusedIterator {}
impl<I> VeclikeIterator for I
where I: Iterator + ExactSizeIterator + DoubleEndedIterator + ::std::iter::FusedIterator {}

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

    pub fn into_csr(self) -> RawCsr<T, R, C>
    where T: AddAssign,
    { self.into_csr_with(AddAssign::add_assign) }

    pub fn into_bee(self) -> RawBee<T, R, C>
    where T: AddAssign,
    { self.into_bee_with(AddAssign::add_assign) }

    pub fn into_indexed_dense(self) -> Indexed<R, Vec<Indexed<C, Vec<T>>>>
    where T: Zero + AddAssign + Clone,
    { self.into_indexed_dense_with(Zero::zero, AddAssign::add_assign) }

    pub fn into_dense(self) -> Vec<Vec<T>>
    where T: Zero + AddAssign + Clone,
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
}

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
    pub fn map<F, T2>(self, mut f: F) -> RawCsr<T2, R, C>
    where F: FnMut(T) -> T2,
    {
        let RawCsr { dim, val, col, row_ptr } = self;
        let val = val.into_iter().map(f).collect();
        RawCsr { dim, val, col, row_ptr }
    }

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

    // Vec to not extend the borrow of self
    pub fn row_ranges(&self) -> Indexed<R, Vec<Range<usize>>> {
        self.row_ptr.raw.windows(2).map(|w| w[0]..w[1]).collect()
    }

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
        RawCoo { dim, row, col, val }
    }
}

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

#[test]
fn bee_into_csr() {
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
    let RawCsr { dim, val, row_ptr, col } = bee.into_csr();
    assert_eq!(dim, (7, 4));
    assert_eq!(val, vec![1.0, 3.0, 2.0]);
    assert_eq!(col, vec![2, 3, 2]);
    assert_eq!(row_ptr.raw, vec![0, 0, 0, 2, 2, 2, 3, 3]);
}
