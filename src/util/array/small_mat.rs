use ::traits::{Field, Ring, IsSquare, IsArray, WithElement};
use ::traits::internal::{PrimitiveRing, PrimitiveFloat};
use ::small_arr::{ArrayFromFunctionExt, ArrayMapExt};
use ::small_arr::{map_arr, try_map_arr, opt_map_arr};

/// Construct a fixed-size matrix from a function on indices.
///
/// Don't stare at the trait bounds for too long, just know that
/// `M` should be a 2D array type, like `[[T; n]; m]`, and the
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
pub fn mat_from_fn<M, F>(mut f: F) -> M
  where
    // FIXME: terrible bounds for public API.
    // (a possible fix is to make another trait, so that we can use as many type
    //  parameters as we want in the impl, using equality constraints.)
    M: ArrayFromFunctionExt,
    M::Element: ArrayFromFunctionExt,
    F: FnMut(usize, usize) -> MatrixElement!{M},
{ M::from_fn(|r| M::Element::from_fn(|c| f(r, c))) }

/// Fallibly construct a fixed-size matrix from a function on indices.
///
/// `M` should be a 2D array type, like `[[T; n]; m]`, and the
/// function must have the signature `fn(row: usize, col: usize) -> Result<T, E>`.
pub fn try_mat_from_fn<M, E, F>(mut f: F) -> Result<M, E>
  where
    M: ArrayFromFunctionExt,
    M::Element: ArrayFromFunctionExt,
    F: FnMut(usize, usize) -> Result<MatrixElement!{M}, E>,
{ M::try_from_fn(|r| M::Element::try_from_fn(|c| f(r, c))) }

/// Fallibly construct a fixed-size matrix from a function on indices.
///
/// `M` should be a 2D array type, like `[[T; n]; m]`, and the
/// function must have the signature `fn(row: usize, col: usize) -> Option<T>`.
pub fn opt_mat_from_fn<M, F>(mut f: F) -> Option<M>
  where
    M: ArrayFromFunctionExt,
    M::Element: ArrayFromFunctionExt,
    F: FnMut(usize, usize) -> Option<MatrixElement!{M}>,
{ M::opt_from_fn(|r| M::Element::opt_from_fn(|c| f(r, c))) }

/// Map a 2D array by value.
///
/// Don't look at the signature too hard, okay? Look at me.
/// ...heyheyhey, I said *look at me!* Everything will be okay,
/// alright? Everything will be okay.
///
/// `M` should be a 2D array type, like `[[A; n]; m]`,
/// and the function must have the signature `fn(A) -> B`.
/// It will produce a value of type `[[B; n]; m]`.
/// Elements are mapped in row major order.
///
/// # Examples
///
/// ```
/// use ::rsp2_array_utils::map_mat;
///
/// let m = [[1, 0], [0, 1]];
///
/// let mat = map_mat(m, |x| 2 * x);
/// assert_eq!(mat, [
///     [2, 0],
///     [0, 2],
/// ]);
/// ```
pub fn map_mat<B, M, F>(m: M, mut f: F) -> Brother!{M, Brother!{M::Element, B}}
  where
    // FIXME: terrible bounds for public API.
    // (a possible fix is to make another trait, so that we can use as many type
    //  parameters as we want in the impl, using equality constraints.)
    M: ArrayMapExt<Brother!{<M as IsArray>::Element, B}>,
    M::Element: ArrayMapExt<B> + WithElement<B>,
    F: FnMut(MatrixElement!{M}) -> B,
{ map_arr(m, |row| map_arr(row, |x| f(x))) }

/// Fallibly map elements of a matrix, short-circuiting on the first `Err(_)`
///
/// `M` should be a 2D array type, like `[[A; n]; m]`,
/// and the function must have the signature `fn(A) -> Result<B, E>`.
/// It will produce a value of type `Result<[[B; n]; m], E>`.
/// Elements are mapped in row major order.
pub fn try_map_mat<B, M, E, F>(m: M, mut f: F) -> Result<Brother!{M, Brother!{M::Element, B}}, E>
where
    M: ArrayMapExt<Brother!{<M as IsArray>::Element, B}>,
    M::Element: ArrayMapExt<B> + WithElement<B>,
    F: FnMut(MatrixElement!{M}) -> Result<B, E>,
{ try_map_arr(m, |row| try_map_arr(row, |x| f(x))) }

/// Fallibly map elements of a matrix, short circuiting on the first `None`.
///
/// `M` should be a 2D array type, like `[[A; n]; m]`,
/// and the function must have the signature `fn(A) -> Option<B>`.
/// It will produce a value of type `Option<[[B; n]; m]>`.
/// Elements are mapped in row major order.
pub fn opt_map_mat<B, M, F>(m: M, mut f: F) -> Option<Brother!{M, Brother!{M::Element, B}}>
where
    M: ArrayMapExt<Brother!{<M as IsArray>::Element, B}>,
    M::Element: ArrayMapExt<B> + WithElement<B>,
    F: FnMut(MatrixElement!{M}) -> Option<B>,
{ opt_map_arr(m, |row| opt_map_arr(row, |x| f(x))) }

/// Extension trait for `<[[T; n]; m]>::determinant()`
///
/// This trait is largely an implementation detail.
/// Prefer the free function `det` instead.
pub trait MatrixDeterminantExt: IsSquare
  where Self::Element2d: Ring
{ fn determinant(&self) -> Self::Element2d; }

/// Extension trait for `<[[T; n]; m]>::inverse()`
///
/// This trait is largely an implementation detail.
/// Prefer the free function `inv` instead.
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

/// Matrix determinant.
pub fn det<M: MatrixDeterminantExt>(mat: &M) -> M::Element2d
where M::Element2d: Ring,
{ mat.determinant() }

/// Matrix inverse.
pub fn inv<M: MatrixInverseExt>(mat: &M) -> M
where M::Element2d: Field,
{ mat.inverse() }
