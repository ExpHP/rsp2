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
        use ::sp2_slice_math::vnormalize;

        let vec: Vec<_> = (0..ndim).map(|_| ::rand::random::<StandardNormal>().0).collect();
        vnormalize(&vec).map(|x| x.0).unwrap_or_else(|_| direction(ndim))
    }
}
