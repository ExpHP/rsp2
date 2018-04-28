#[macro_use]
extern crate failure;
extern crate rand;
extern crate lapacke;
extern crate slice_of_array;
#[macro_use]
extern crate ndarray;
#[cfg_attr(test, macro_use)]
extern crate rsp2_assert_close;
extern crate lapack_src;

use ::failure::{Error};
use ::ndarray::{Array, Array2, ArrayView2, ArrayBase, Ix2, Axis};

#[derive(Debug, Fail)]
#[fail(display = "matrix was perfectly degenerate")]
pub struct DegenerateMatrixError;

pub fn dot<A, B>(a: &A, b: &B) -> <A as ndarray::linalg::Dot<B>>::Output
where A: ndarray::linalg::Dot<B>
{ a.dot(b) }

use self::c_matrix::CMatrix;
mod c_matrix {
    use super::*;

    /// Owned, contiguous, C-order matrix data.
    #[derive(Debug, Clone)]
    pub struct CMatrix<A = f64>(Array2<A>);

    impl<A> CMatrix<A> {
        pub fn into_inner(self) -> Array2<A> { self.0 }
        pub fn c_order_data(&self) -> &[A] { self.0.as_slice().expect("(BUG) not c-order!!") }
        pub fn c_order_data_mut(&mut self) -> &mut [A] { self.0.as_slice_mut().expect("(BUG) not c-order!!") }
        pub fn stride(&self) -> usize { self.cols() }
    }

    impl<A> ::std::ops::Deref for CMatrix<A> {
        type Target = Array2<A>;

        fn deref(&self) -> &Self::Target { &self.0 }
    }

    impl<A: Clone> From<Array2<A>> for CMatrix<A> {
        fn from(arr: Array2<A>) -> Self {
            if arr.is_standard_layout() {
                CMatrix(arr)
            } else {
                arr.view().into()
            }
        }
    }

    impl<'a, A: Clone> From<ArrayView2<'a, A>> for CMatrix<A>
    {
        fn from(arr: ArrayView2<'a, A>) -> Self {
            let dim = arr.raw_dim();
            if let Some(data) = arr.as_slice() {
                CMatrix(Array::from_shape_vec(dim, data.to_vec()).expect("BUG"))
            } else {
                CMatrix(Array::from_shape_vec(dim, arr.iter().cloned().collect()).expect("BUG"))
            }
        }
    }

    impl<'a, A: Clone, S> From<&'a ArrayBase<S, Ix2>> for CMatrix<A>
    where S: ndarray::Data<Elem = A>,
    {
        fn from(arr: &'a ArrayBase<S, Ix2>) -> Self {
            arr.view().into()
        }
    }

    #[test]
    fn test_into_c_matrix() {

        let check = |arr: Array2<_>, expected| {
            let c_mat_ref = CMatrix::from(&arr);
            let c_mat_view = CMatrix::from(arr.view());
            let c_mat_own = CMatrix::from(arr);
            assert_eq!(c_mat_ref.c_order_data(), expected);
            assert_eq!(c_mat_view.c_order_data(), expected);
            assert_eq!(c_mat_own.c_order_data(), expected);
        };

        use ndarray::ShapeBuilder;

        // exercise the standard layout code paths
        let arr = Array::from_shape_vec((2, 3), vec![00, 01, 02, 10, 11, 12]).unwrap();
        assert!(arr.is_standard_layout());
        assert!(arr.as_slice().is_some());
        check(arr, &[00, 01, 02, 10, 11, 12]);

        // non-standard layout but contiguous (`to_owned()` would burn us here)
        let arr = Array::from_shape_vec((2, 3).f(), vec![00, 10, 01, 11, 02, 12]).unwrap();
        assert!(!arr.is_standard_layout());
        assert!(arr.as_slice().is_none());
        check(arr, &[00, 01, 02, 10, 11, 12]);

        // non-standard layout, non-contiguous
        let mut arr = Array::from_shape_vec((2, 4), vec![03, 00, 01, 02, 13, 10, 11, 12]).unwrap();
        arr.slice_axis_inplace(Axis(1), (1..).into()); // O_o discontiguous owned arrays are a thing
        check(arr, &[00, 01, 02, 10, 11, 12]);
    }
}

pub type ArrayBase2<S> = ArrayBase<S, ndarray::Ix2>;

pub fn left_pseudoinverse(mat: &ArrayView2<f64>) -> Result<Array2<f64>, Error>
{
    // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=160
    //
    // I've observed this strategy to be *far* more accurate than attempting
    // to use dgesv to solve  'mat mat.T P.T = mat'
    //
    // NOTE: for further possible improvement, see also thombos' quote:
    //
    // > I now decided to just calculate the reduced svd using cgesdd (or cgesvd) and explicitely
    // > calculate the inverse V*(S^-1)*U^H, which is in my case 100 times faster than cgelss_ .
    // > Its stunning how big the difference in timing is. Is this normal ?
    // > The matrices to pseudo-invert are 15000x640.
    //
    // (I don't think we have to worry about this for computing force constants, because
    //  our matrices tend to be around size... I'm not sure.  6x3?  In any case, they don't
    //  scale with the size of the system (we simply end up with more of them to solve))
    let max = usize::max(mat.rows(), mat.cols());
    let b = Array2::<f64>::eye(max);
    let b = lapacke_least_squares_svd(mat.into(), b.into())?;

    Ok(b.slice(s![..mat.cols(), ..mat.rows()]).to_owned())
}

/// Solves `output = square * rhs` using LAPACKe's dgesv.
pub fn lapacke_linear_solve(mut square: CMatrix, mut rhs: CMatrix) -> Result<CMatrix, DegenerateMatrixError> {
    assert!(square.is_square());
    assert_eq!(square.cols(), rhs.rows());

    let layout = ::lapacke::Layout::RowMajor;

    let n = rhs.rows() as i32;
    let nrhs = rhs.cols() as i32;
    let lda = square.stride() as i32;
    let ldb = rhs.stride() as i32;

    {
        // lapacke hates size-zero arrays.
        let a = match square.len() {
            0 => return Ok(rhs), // rhs must also have zero size; trivial solution
            _ => square.c_order_data_mut(),
        };
        let b = match rhs.len() {
            0 => return Ok(rhs), // trivial solution
            _ => rhs.c_order_data_mut(),
        };

        let mut ipiv = vec![0; n as usize];
        let ipiv = &mut ipiv;
        assert!(ipiv.len() > 0);

        match unsafe { ::lapacke::dgesv(layout, n, nrhs, a, lda, ipiv, b, ldb) } {
            0 => { /* okey dokey */ },
            info if info < 0 => panic!("bad arg number {} to dgesv", -info),
            _info => return Err(DegenerateMatrixError),
        }
    } // end borrows

    Ok(rhs)
}

/// Minimizes 2-norm of `matrix * x - rhs` using LAPACKe's dgelss.
pub fn lapacke_least_squares_svd(mut matrix: CMatrix, mut rhs: CMatrix) -> Result<Array2<f64>, Error> {
    assert!(matrix.cols() <= rhs.rows());

    let layout = ::lapacke::Layout::RowMajor;

    let m = matrix.rows() as i32;
    let n = matrix.cols() as i32;
    let nrhs = rhs.cols() as i32;
    let lda = matrix.stride() as i32;
    let ldb = rhs.stride() as i32;

    let rcond = -1f64; // use machine precision

    // lapacke hates size-zero arrays.
    assert_ne!(ldb, 0, "TODO"); // I think this is trivial? (return rhs)
    assert_ne!(lda, 0, "TODO"); // I think this is underdetermined/singular when ldb != 0?
    {
        let a = matrix.c_order_data_mut();
        let b = rhs.c_order_data_mut();

        let mut s = vec![0f64; i32::min(m, n) as usize];
        let s = &mut s;

        let mut rank = 0;
        let rank = &mut rank;

        match unsafe { ::lapacke::dgelss(layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank) } {
            0 => { /* okey dokey */ },
            info if info < 0 => panic!("bad arg number {} to dgelss", -info),
            info => bail!("error during SVD ({} non-converging elements)", info),
        }
    } // end borrows

    Ok(rhs.slice_axis(Axis(0), (..n as usize).into()).to_owned())
}

#[test]
fn test_pseudoinverse() {
    use ::rand::Rng;

//    for _ in 0.. { // stress test
    for _ in 0..100 {
        let mut rng = ::rand::thread_rng();
        // Produce an overdetermined or well-determined problem to solve.
        let r = rng.gen_range(1, 20);
        let c = rng.gen_range(1, r + 1);

        let mat = Array2::from_shape_fn((r, c), |_| 1.0 - 2.0 * rng.gen::<f64>());
        let p_inv = match left_pseudoinverse(&mat.view()) {
            Ok(inv) => inv,
            Err(_) => panic!("singular random uniform? now that's what I call bad luck {:?}", (r, c)),
        };
        let prod = dot(&p_inv, &mat);
        for i in 0..c {
            for j in 0..c {
                // NOTE: I've seen up to 1.2e-10 absolute error on a long run
                if i == j {
                    assert_close!(abs=1e-8, 1.0, prod[(i, j)], "{:#?}", prod);
                } else {
                    assert_close!(abs=1e-8, 0.0, prod[(i, j)], "{:#?}", prod);
                }
            }
        }
    }
}
