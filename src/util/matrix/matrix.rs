use ::rsp2_array_types::V3;
use ::std::ops::{Index, IndexMut};
use ::std::ops::{RangeTo};
use ::std::marker::PhantomData;
use ::slice_of_array::prelude::*;
use ::rsp2_shims::range_bounds::{RangeBounds, RangeBoundsHelper};

/// Owned matrix type with C layout.
// please resist the urge to go n-dimensional
#[derive(Debug, Clone)]
pub struct Matrix<T = f64> {
    // c-contiguous, row-contiguous data
    data: Vec<T>,
    // invariant: height * width == data.len()
    height: usize,
    width: usize,
}

// strictly speaking we should be working with a raw pointer
// and PhantomData slices because we do not want to have overlapping
// &mut, but I'll hold off on that whole song and dance for now
pub struct MatrixRef_<T, Vs> {
    slice: Vs,
    dims: (usize, usize),
    strides: (usize, usize),
    _dummy: PhantomData<T>, // constrain some impls
}
pub type MatrixRef<'a, T = f64> = MatrixRef_<T, &'a [T]>;
pub type MatrixMut<'a, T = f64> = MatrixRef_<T, &'a mut [T]>;

pub trait ToOwnedMatrix<T> {
    fn to_owned_matrix(&self) -> Matrix<T>;
}
pub trait AsMatrixRef<T> {
    fn as_matrix_ref(&self) -> MatrixRef<T>;
}
pub trait AsMatrixMut<T> {
    fn as_matrix_mut(&mut self) -> MatrixMut<T>;
}

pub type ContiguousRows<'a, T> = ::std::slice::Chunks<'a, T>;
pub type ContiguousRowsMut<'a, T> = ::std::slice::ChunksMut<'a, T>;

impl<T> Matrix<T> {
    pub fn from_row_major_data((height, width): (usize, usize), data: Vec<T>) -> Self
    {
        assert_eq!(data.len(), height * width);
        Matrix { data, height, width }
    }

    // for making implementation DRY
    fn get_matrix_ref_info(&self) -> MatrixRef_<(), ()>
    { MatrixRef_ {
        slice: (),
        _dummy: (),
        strides: (self.width, 1),
        dims: (self.height, self.width),
    }}
}

impl<T> AsMatrixRef<T> for Matrix<T> {
    fn as_matrix_ref(&self) -> MatrixRef<T> {
        let MatrixRef_ { dims, strides, .. } = self.get_matrix_ref_info();
        let slice = &self.data;
        MatrixRef_ { slice, dims, strides, _dummy: PhantomData }
    }
}
impl<T> AsMatrixMut<T> for Matrix<T> {
    fn as_matrix_mut(&mut self) -> MatrixMut<T> {
        let MatrixRef_ { dims, strides, .. } = self.get_matrix_ref_info();
        let slice = &mut self.data;
        MatrixRef_ { slice, dims, strides, _dummy: PhantomData }
    }
}
impl<T, Vs: AsRef<[T]>> AsMatrixRef<T> for MatrixRef_<T, Vs> {
    fn as_matrix_ref(&self) -> MatrixRef<T> {
        let MatrixRef_ { dims, strides, .. } = *self;
        let slice = self.slice.as_ref();
        MatrixRef_ { slice, dims, strides, _dummy: PhantomData }
    }
}
impl<T, Vs: AsMut<[T]>> AsMatrixMut<T> for MatrixRef_<T, Vs> {
    fn as_matrix_mut(&mut self) -> MatrixMut<T> {
        let MatrixRef_ { dims, strides, .. } = *self;
        let slice = self.slice.as_mut();
        MatrixRef_ { slice, dims, strides, _dummy: PhantomData }
    }
}

impl<T> Matrix<T> {
    pub fn row_major_data(&self) -> &[T] { self.data.as_ref() }
    pub fn row_major_data_mut(&mut self) -> &mut [T] { &mut self.data }
    pub fn rows(&self) -> ContiguousRows<T> { self.data.chunks(self.width) }
    pub fn rows_mut(&mut self) -> ContiguousRowsMut<T> { self.data.chunks_mut(self.width) }
}

impl<'a, T> AsMatrixRefExt<T> for MatrixRef<'a, T> {
    fn strides(&self) -> (usize, usize) { self.strides }
    fn dims(&self) -> (usize, usize) { self.dims }
    fn row_stride(&self) -> usize { self.strides.0 }
    fn col_stride(&self) -> usize { self.strides.1 }
    fn num_rows(&self) -> usize { self.dims.0 }
    fn num_cols(&self) -> usize { self.dims.1 }
    fn is_square(&self) -> bool { self.dims.0 == self.dims.1 }
    fn size(&self) -> usize { self.dims.0 * self.dims.1 }
    fn transpose(&self) -> MatrixRef<T>
    {
        let MatrixRef_ { ref slice, dims, strides, _dummy } = *self;
        let dims = (dims.1, dims.0);
        let strides = (strides.1, strides.0);
        MatrixRef_ { slice, dims, strides, _dummy }
    }
    fn slice_rows<R: RangeBounds<usize>>(&self, range: R) -> MatrixRef<T>
    {
        let range = range.to_range(self.num_rows());
        let dims = (range.len(), self.dims.1);
        let strides = self.strides;
        let slice = &self.slice[range.start * self.row_stride()..];
        MatrixRef_ { slice, dims, strides, _dummy: PhantomData }
    }
}

impl<'a, T> AsMatrixMutExt<T> for MatrixMut<'a, T> {
    fn transpose_mut(&mut self) -> MatrixMut<T>
    {
        let MatrixRef_ { ref mut slice, dims, strides, _dummy } = *self;
        let dims = (dims.1, dims.0);
        let strides = (strides.1, strides.0);
        MatrixRef_ { slice, dims, strides, _dummy }
    }
    fn slice_rows_mut<R: RangeBounds<usize>>(&mut self, range: R) -> MatrixMut<T>
    {
        let range = range.to_range(self.num_rows());
        let dims = (range.len(), self.dims.1);
        let strides = self.strides;
        let slice = &mut self.slice[range.start * self.row_stride()..];
        MatrixRef_ { slice, dims, strides, _dummy: PhantomData }
    }
}

// FIXME all should be inherent methods via macros
pub trait AsMatrixRefExt<T>: AsMatrixRef<T>
{
    // NOTE: Please add methods to the impl for MatrixRef before adding them here
    //       to make sure you don't forget to override the default definitions
    fn strides(&self) -> (usize, usize) { self.as_matrix_ref().strides() }
    fn row_stride(&self) -> usize { self.as_matrix_ref().row_stride() }
    fn col_stride(&self) -> usize { self.as_matrix_ref().col_stride() }
    fn dims(&self) -> (usize, usize) { self.as_matrix_ref().dims() }
    fn num_rows(&self) -> usize { self.as_matrix_ref().num_rows() }
    fn num_cols(&self) -> usize { self.as_matrix_ref().num_cols() }
    fn is_square(&self) -> bool { self.as_matrix_ref().is_square() }
    fn size(&self) -> usize { self.as_matrix_ref().size() }
    fn slice_rows<R: RangeBounds<usize>>(&self, range: R) -> MatrixRef<T> { self.as_matrix_ref().slice_rows(range) }
    fn transpose(&self) -> MatrixRef<T> { self.as_matrix_ref().transpose() }
}
impl<T> AsMatrixRefExt<T> for Matrix<T> {}
impl<'a, T> AsMatrixRefExt<T> for MatrixMut<'a, T> {}

// FIXME all should be inherent methods via macros
pub trait AsMatrixMutExt<T>: AsMatrixMut<T>
{
    // NOTE: Please add methods to the impl for MatrixMut before adding them here
    //       to make sure you don't forget to override the default definitions
    fn slice_rows_mut<R: RangeBounds<usize>>(&mut self, range: R) -> MatrixMut<T> { self.as_matrix_mut().slice_rows_mut(range) }
    fn transpose_mut(&mut self) -> MatrixMut<T> { self.as_matrix_mut().transpose_mut() }
}
impl<T> AsMatrixMutExt<T> for Matrix<T> {}

pub fn is_matmul_compatible<T, U>(a: &MatrixRef<T>, b: &MatrixRef<U>) -> bool
{ a.num_cols() == b.num_rows() }

impl<T: Clone> Matrix<T> {
    pub fn new_filled((height, width): (usize, usize), fill: &T) -> Self
    { Matrix {
        data: (0..height * width).map(|_| fill.clone()).collect(),
        height,
        width,
    }}

    pub fn to_transpose(&self) -> Self
    {
        let mut data = Vec::with_capacity(self.data.len());
        for c in 0..self.num_cols() {
            for r in 0..self.num_rows() {
                data.push(self[(r, c)].clone());
            }
        }
        Matrix { data, height: self.width, width: self.height }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    #[inline(always)] // inlining should often remove bounds checks
    fn index(&self, index: (usize, usize)) -> &Self::Output
    { &self.as_matrix_ref()[index] }
}

impl<'a, T> Index<(usize, usize)> for MatrixRef<'a, T> {
    type Output = T;

    #[inline(always)] // inlining should often remove bounds checks
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output
    { &self.slice[r * self.strides.0 + c * self.strides.1] }
}

impl<'a, T: Clone + 'a> From<&'a [V3<T>]> for Matrix<T> {
    fn from(it: &'a [V3<T>]) -> Self
    { Matrix {
        data: it.flt().into(),
        height: it.len(),
        width: 3,
    }}
}

impl<'a, 'b> ::std::ops::Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Matrix
    { matmul(self, rhs) }
}

fn matmul(a: &Matrix, b: &Matrix) -> Matrix
{
    assert_eq!(a.num_cols(), b.num_rows());

    // this is suboptimal.  who cares.
    let mut out = Matrix::new_filled((a.num_rows(), b.num_cols()), &f64::from_bits(0xf00d));
    let b_t = b.to_transpose();
    for (out_row, a_row) in ::util::zip_eq(out.rows_mut(), a.rows()) {
        for (out, b_col) in ::util::zip_eq(out_row, b_t.rows()) {
            for (x, y) in ::util::zip_eq(a_row, b_col) {
                *out += x * y;
            }
        }
    }
    out
}
