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

macro_rules! zip_eq {
    ($a:expr $(,)*) => {
        $a.into_iter().map(|a| (a,))
    };
    ($a:expr, $b:expr $(,)*) => {
        ::util::zip_eq($a, $b)
    };
    ($a:expr, $b:expr, $c:expr $(,)*) => {
        ::util::zip_eq(::util::zip_eq($a, $b), $c)
            .map(|((a, b), c)| (a, b, c))
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)*) => {
        ::util::zip_eq(::util::zip_eq(::util::zip_eq($a, $b), $c), $d)
            .map(|(((a, b), c), d)| (a, b, c, d))
    };
}

pub fn zip_eq<As, Bs>(a: As, b: Bs) -> ::std::iter::Zip<As::IntoIter, Bs::IntoIter>
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

pub fn partial_max<T: PartialOrd>(it: impl IntoIterator<Item=T>) -> Option<T> {
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
