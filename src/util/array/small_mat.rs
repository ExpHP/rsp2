use ::traits::{Field, Ring, IsSquare};
use ::traits::internal::{PrimitiveRing, PrimitiveFloat};
use ::small_vec::ArrayFromFunctionExt;

/// Construct a fixed-size matrix from a function on indices.
///
/// Don't stare at the trait bounds for too long, just know that
/// `V` should be a 2D array type, like `[[T; n]; m]`, and the
/// function must have the signature `fn(row: usize, col: usize) -> T`.
///
/// # Examples
///
/// ```
/// use ::rsp2_array_utils::mat_from_fn;
///
/// let x = [2i32, 3];
///
/// let mat: [[_; 4]; 2] = mat_from_fn(|r,c| x[r].pow(c as u32));
/// assert_eq!(mat, [
///     [1, 2, 4,  8],
///     [1, 3, 9, 27],
/// ]);
/// ```
pub fn mat_from_fn<V, F>(mut f: F) -> V
  where
    // FIXME: terrible bounds for public API
    V: ArrayFromFunctionExt + Default,
    V::Element: ArrayFromFunctionExt + Default,
    F: FnMut(usize, usize) -> <V::Element as ::traits::IsArray>::Element,
{ V::from_fn(|r| V::Element::from_fn(|c| f(r, c))) }

/// Extension trait for `<[[T; n]; m]>::determinant()`
pub trait MatrixDeterminantExt: IsSquare
  where Self::Element2d: Ring
{ fn determinant(&self) -> Self::Element2d; }

/// Extension trait for `<[[T; n]; m]>::inverse()`
pub trait MatrixInverseExt: IsSquare
  where Self::Element2d: Field
{ fn inverse(&self) -> Self; }


impl<T: Ring> MatrixDeterminantExt for nd![T; 2; 2]
where T: PrimitiveRing,
{
    fn determinant(&self) -> T {
        self[0][0] * self[1][1] - self[0][1] * self[1][0]
    }
}

impl<T: Ring> MatrixDeterminantExt for nd![T; 3; 3]
where T: PrimitiveRing,
{
    fn determinant(&self) -> T { 
        let destructure = |v: &[T; 3]| { (v[0], v[1], v[2]) };
        let (a0, a1, a2) = destructure(&self[0]);
        let (b0, b1, b2) = destructure(&self[1]);
        let (c0, c1, c2) = destructure(&self[2]);
        
        T::zero()
        + a0 * b1 * c2
        + a1 * b2 * c0
        + a2 * b0 * c1
        - a0 * b2 * c1
        - a1 * b0 * c2
        - a2 * b1 * c0
     }
}


impl<T: Field> MatrixInverseExt for nd![T; 2; 2]
  where T: PrimitiveFloat,
{
    fn inverse(&self) -> Self {
        let rdet = T::one() / self.determinant();
        [ [ self[1][1] * rdet, -self[0][1] * rdet]
        , [-self[1][0] * rdet,  self[0][0] * rdet]
        ]
    }
}

impl<T: Field> MatrixInverseExt for nd![T; 3; 3]
  where T: PrimitiveFloat,
{
    fn inverse(&self) -> Self {
        let cofactors: nd![_; 3; 3] = mat_from_fn(|r, c|
            T::zero()
            + self[(r+1) % 3][(c+1) % 3] * self[(r+2) % 3][(c+2) % 3]
            - self[(r+1) % 3][(c+2) % 3] * self[(r+2) % 3][(c+1) % 3]
        );
        let det = (0..3).map(|i| self[0][i] * cofactors[0][i]).sum::<T>();
        let rdet = T::one() / det;
        mat_from_fn(|r, c| rdet * cofactors[c][r])
    }
}

#[cfg(test)]
mod tests {
    use super::MatrixInverseExt;

    #[test]
    fn test_inverse_2() {
        let actual = [[7., 2.], [-11., 4.]].inverse();
        let expected = [
            [ 2./25., -1./25.],
            [11./50.,  7./50.],
        ];

        for r in 0..2 {
            for c in 0..2 {
                assert_close!(abs=1e-12, expected[r][c], actual[r][c]);
            }
        }
    }

    #[test]
    fn test_inverse_3() {
        let actual = [[1., 2., 4.], [5., 2., 1.], [3., 6., 3.]].inverse();
        let expected = [
            [ 0./1.,  1./4., -1./12.],
            [-1./6., -1./8., 19./72.],
            [ 1./3.,  0./1., -1./9. ],
        ];

        for r in 0..3 {
            for c in 0..3 {
                assert_close!(abs=1e-12, expected[r][c], actual[r][c]);
            }
        }
    }
}
