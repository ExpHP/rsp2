macro_rules! zip_eq {
    ($a:expr $(,)*) => {
        $a.into_iter().map(|a| (a,))
    };
    ($a:expr, $b:expr $(,)*) => {
        ::shared::util::zip_eq($a, $b)
    };
    ($a:expr, $b:expr, $c:expr $(,)*) => {
        ::shared::util::zip_eq(::shared::util::zip_eq($a, $b), $c)
            .map(|((a, b), c)| (a, b, c))
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)*) => {
        ::shared::util::zip_eq(::shared::util::zip_eq(::shared::util::zip_eq($a, $b), $c), $d)
            .map(|(((a, b), c), d)| (a, b, c, d))
    };
}

pub(crate) fn zip_eq<As, Bs>(a: As, b: Bs) -> ::std::iter::Zip<As::IntoIter, Bs::IntoIter>
where
    As: IntoIterator, As::IntoIter: ExactSizeIterator,
    Bs: IntoIterator, Bs::IntoIter: ExactSizeIterator,
{
    let (a, b) = (a.into_iter(), b.into_iter());
    assert_eq!(a.len(), b.len());
    a.zip(b)
}

macro_rules! impl_newtype_debug {
    ($Type:ident) => {
        impl ::std::fmt::Debug for $Type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                ::std::fmt::Debug::fmt(&self.0, f)
            }
        }
    };
}

pub(crate) fn partial_max<T: PartialOrd>(it: impl IntoIterator<Item=T>) -> Option<T> {
    let mut it = it.into_iter();
    let first = it.next()?;
    Some(it.fold(first, |acc, b| {
        if acc < b {
            b
        } else {
            acc
        }
    }))
}
