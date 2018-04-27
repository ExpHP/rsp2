use ::{Result, ErrorKind};
use super::Matrix;

pub fn left_pseudoinverse(mat: &Matrix) -> Result<Matrix>
{
    // mat * mat.T
    // each element (row, col) is the row'th 3-vector dotted by the col'th 3-vector
    let big_square = mat * &mat.to_transpose();

    // (mat * mat.T).T
    let big_square_t = big_square; // it's symmetric

    // (mat * mat.T).T^-1 mat
    let solved = lapacke_linear_solve(big_square_t, mat.clone())?;

    // mat.T * (mat * mat.T)^-1
    Ok(solved.to_transpose())
}

/// Solves `output = square * rhs` using LAPACKe's dgesv.
pub fn lapacke_linear_solve(mut square: Matrix, mut rhs: Matrix) -> Result<Matrix> {
    assert!(square.is_square());
    assert_eq!(square.num_cols(), rhs.num_rows());

    let layout = ::lapacke::Layout::RowMajor;

    let n = rhs.num_rows() as i32;
    let nrhs = rhs.num_cols() as i32;
    let lda = square.row_stride() as i32;
    let ldb = rhs.row_stride() as i32;

    {
        // lapacke hates size-zero arrays.
        let a = match square.size() {
            0 => return Ok(rhs), // rhs must also have zero size; trivial solution
            _ => square.row_major_data_mut(),
        };
        let b = match rhs.size() {
            0 => return Ok(rhs), // trivial solution
            _ => rhs.row_major_data_mut(),
        };

        let mut ipiv = vec![0; n as usize];
        let ipiv = &mut ipiv;
        assert!(ipiv.len() > 0);

        match unsafe { ::lapacke::dgesv(layout, n, nrhs, a, lda, ipiv, b, ldb) } {
            0 => { /* okey dokey */ },
            info if info < 0 => panic!("bad arg number {} to dgesv", -info),
            info => bail!(ErrorKind::SingularMatrix),
        }
    } // end borrows

    Ok(rhs)
}

/// Minimizes 2-norm of `matrix * x - rhs` using LAPACKe's dgelss.
pub fn lapacke_least_squares_svd(mut matrix: Matrix, mut rhs: Matrix) -> Result<Matrix> {
    assert!(matrix.num_cols() <= rhs.num_rows());

    let layout = ::lapacke::Layout::RowMajor;

    let m = matrix.num_rows() as i32;
    let n = matrix.num_cols() as i32;
    let nrhs = rhs.num_cols() as i32;
    let lda = matrix.row_stride() as i32;
    let ldb = rhs.row_stride() as i32;

    let rcond = -1f64; // use machine precision

    // lapacke hates size-zero arrays.
    assert_ne!(ldb, 0, "TODO"); // I think this is trivial? (return rhs)
    assert_ne!(lda, 0, "TODO"); // I think this is underdetermined/singular when ldb != 0?
    {
        let a = matrix.row_major_data_mut();
        let b = rhs.row_major_data_mut();

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

    rhs.take_cols(n)
}

#[test]
fn test_pseudoinverse() {
    use ::rand::Rng;

    for _ in 0..10000 {
        let mut rng = ::rand::thread_rng();
        let r = rng.gen_range(10, 20);
        let c = rng.gen_range(2, 3);
        let data = rng.gen_iter().map(|x: f64| 1.0 - 2.0 * x).take(r*c).collect();

        let mat = Matrix::from_row_major_data((r, c), data);
        let p_inv = match left_pseudoinverse(&mat) {
            Ok(inv) => inv,
            Err(_) => panic!("singular random uniform? now that's what I call bad luck {:?}", (r, c)),
        };
        let prod = &p_inv * &mat;
        for i in 0..c {
            for j in 0..c {
                if i == j {
                    assert_close!(rel=1e-10, 1.0, prod[(i, j)], "{:#?}", prod);
                } else {
                    assert_close!(abs=1e-10, 0.0, prod[(i, j)], "{:#?}", prod);
                }
            }
        }
    }
}
