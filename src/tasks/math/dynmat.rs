use ::{Result, ErrorKind};
use ::rsp2_array_types::{V3, M33, dot};
use ::rsp2_structure::{FracOp, Perm};

#[derive(Debug, Clone, Default)]
pub struct ForceSets {
    atom_displaced: Vec<u32>,
    atom_affected: Vec<u32>,
    cart_force: Vec<V3>,
    cart_displacement: Vec<V3>,
}

impl ForceSets {
    // `perm` should describe the symmop as a permutation, such that applying the
    // operator moves the atom at `coords[i]` to `coords[perm[i]]`
    fn derive_from_symmetry(&self, cart_rot: &M33, perm: &Perm) -> Self {
        let atom_displaced = self.atom_displaced.iter().map(|&i| perm[i]).collect();
        let atom_affected = self.atom_affected.iter().map(|&i| perm[i]).collect();
        let cart_op_t = cart_rot.t();
        let cart_force = self.cart_force.iter().map(|v| v * &cart_op_t).collect();
        let cart_displacement = self.cart_displacement.iter().map(|v| v * &cart_op_t).collect();
        ForceSets { atom_displaced, atom_affected, cart_force, cart_displacement }
    }

    fn concat_from<Ss>(iter: Ss) -> Self
    where Ss: IntoIterator<Item=ForceSets>,
    {
        iter.into_iter().fold(Self::default(), |mut a, b| {
            a.atom_affected.extend(b.atom_affected);
            a.atom_displaced.extend(b.atom_displaced);
            a.cart_force.extend(b.cart_force);
            a.cart_displacement.extend(b.cart_displacement);
            a
        })
    }

    fn solve_force_constants() -> ForceConstants
    {
        // build a (likely overconstrained) system of equations for each interacting (i,j) pair
        unimplemented!()
        // solve using pseudoinverse
    }
}

pub struct ForceConstants {
    row_atom: Vec<usize>,
    col_atom: Vec<usize>,
    // this might be awkward to work with...
    cart_matrix: Vec<M33>,
}

// rsp2 doesn't frequently work with matrices that are variable in more
// than one dimension, so this is kind of defined on the spot
//
// if it's later worth reusing somewhere else, move it to somewhere better
mod matrix {
    use super::*;

    pub struct Matrix<T> {
        data: Vec<T>, // c-contiguous, row-contiguous
        // invariant:  width * height == data.len()
        // Both are stored for the sake of the degenerate case where one dimension is zero.
        height: usize,
        width: usize,
    }

    impl<T> Matrix<T> {
        fn new_filled((height, width): (usize, usize), fill: &T) -> Self
        where T: Clone
        {
            let data = (0..height * width).map(|_| fill.clone()).collect();
            Matrix { data, height, width }
        }
        fn num_rows(&self) -> usize { self.height }
        fn num_cols(&self) -> usize { self.width }
        fn row_major_data(&self) -> &[T] { &self.data }
        fn row_major_data_mut(&mut self) -> &mut [T] { &mut self.data }
        fn is_square(&self) -> bool { self.width == self.height }
        fn size(&self) -> usize { self.data.len() }
    }

    pub fn pseudoinverse(mat: Vec<V3>) -> Matrix<f64> {
        // mat * mat.T
        // each element (row, col) is the row'th 3-vector dotted by the col'th 3-vector
        let dim = mat.len();
        let big_square = {
            let mut data = vec![0.0; dim * dim];
            {
                let chunks: Vec<_> = data.chunks_mut(dim).collect();
                for (data, row_v3) in ::util::zip_eq(chunks, &mat) {
                    for (data, col_v3) in ::util::zip_eq(data, &mat) {
                        *data = dot(row_v3, col_v3);
                    }
                }
            }
            Matrix { data, width: dim, height: dim }
        };

        panic!()
    }

    /// Solves `output = square * rhs` using LAPACKe's dgesv.
    fn lapacke_linear_solve(mut square: Matrix<f64>, mut rhs: Matrix<f64>) -> Result<Matrix<f64>> {
        use ::lapacke::{Layout, dgesv};
        assert!(square.is_square());
        assert_eq!(square.num_cols(), rhs.num_rows());

        let layout = Layout::RowMajor;

        let n = rhs.num_rows() as i32;
        let nrhs = rhs.num_cols() as i32;
        let lda = square.num_cols() as i32;
        let ldb = rhs.num_cols() as i32;

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

            match unsafe { dgesv(layout, n, nrhs, a, lda, ipiv, b, ldb) } {
                0 => { /* okey dokey */ },
                info if info < 0 => panic!("bad arg number {} to dgesv", -info),
                info => bail!(ErrorKind::SingularMatrix),
            }
        } // end borrows

        Ok(rhs)
    }
}




