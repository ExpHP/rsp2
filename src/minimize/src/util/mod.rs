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

/// General-use empty type, until `!` gets stabilized.
///
/// See https://github.com/rust-lang/rfcs/blob/master/text/1216-bang-type.md
///
/// This doesn't come with the implicit conversions.
/// For now, you'll need to explicitly `match` on it or implement `From<Never>`.
#[derive(Serialize, Deserialize)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum Never { }

pub(crate) mod cache;

#[allow(dead_code)]
pub(crate) mod random {
    pub(crate) fn uniform(a: f64, b: f64) -> f64 {
        a + ::rand::random::<f64>() * (b - a)
    }

    pub(crate) fn uniform_n(ndim: usize, a: f64, b: f64) -> Vec<f64> {
        (0..ndim).map(|_| uniform(a, b)).collect()
    }

    pub(crate) fn uniform_box(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len());
        (0..a.len()).map(|i| uniform(a[i], b[i])).collect()
    }

    pub(crate) fn direction(ndim: usize) -> Vec<f64> {
        use ::rand::distributions::normal::StandardNormal;
        use ::rsp2_slice_math::vnormalize;

        let vec: Vec<_> = (0..ndim).map(|_| ::rand::random::<StandardNormal>().0).collect();
        vnormalize(&vec).map(|x| x.0).unwrap_or_else(|_| direction(ndim))
    }
}
