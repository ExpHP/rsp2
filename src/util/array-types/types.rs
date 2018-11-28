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

use std::ops::{Deref, DerefMut};
use std::fmt;

// ---------------------------------------------------------------------------

/// A 2-dimensional vector with operations for linear algebra.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct V2<X=f64>(pub [X; 2]);

/// A 3-dimensional vector with operations for linear algebra.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct V3<X=f64>(pub [X; 3]);

/// A 4-dimensional vector with operations for linear algebra.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct V4<X=f64>(pub [X; 4]);

// ---------------------------------------------------------------------------

/// A linear algebra dense matrix with 2 rows and fixed width.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct M2<V>(pub [V; 2]);

/// A linear algebra dense matrix with 3 rows and fixed width.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct M3<V>(pub [V; 3]);

/// A linear algebra dense matrix with 4 rows and fixed width.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct M4<V>(pub [V; 4]);

/// A square dense 2x2 matrix.
pub type M22<X=f64> = M2<V2<X>>;
/// A square dense 3x3 matrix.
pub type M33<X=f64> = M3<V3<X>>;
/// A square dense 4x4 matrix.
pub type M44<X=f64> = M4<V4<X>>;

// ---------------------------------------------------------------------------
// All types behave generally like their backing array type.

pub type Iter<'a, X> = std::slice::Iter<'a, X>;
pub type IterMut<'a, X> = std::slice::IterMut<'a, X>;

gen_each!{
    [
        {V2 X 2} {V3 X 3} {V4 X 4}
        {M2 V 2} {M3 V 3} {M4 V 4}
    ]
    for_each!(
        {$Cn:ident $T:ident $n:tt}
    ) => {
        impl<$T> Deref for $Cn<$T> {
            type Target = [$T; $n];

            #[inline(always)]
            fn deref(&self) -> &Self::Target
            { &self.0 }
        }

        impl<$T> DerefMut for $Cn<$T> {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target
            { &mut self.0 }
        }

        // Fix a paper cut not solved by Deref, which is that many methods
        // take `I: IntoIterator`.
        impl<'a, $T> IntoIterator for &'a $Cn<$T> {
            type Item = &'a $T;
            type IntoIter = Iter<'a, $T>;

            #[inline(always)]
            fn into_iter(self) -> Self::IntoIter
            { self.0.iter() }
        }

        impl<'a, $T> IntoIterator for &'a mut $Cn<$T> {
            type Item = &'a mut $T;
            type IntoIter = IterMut<'a, $T>;

            #[inline(always)]
            fn into_iter(self) -> Self::IntoIter
            { self.0.iter_mut() }
        }

        // forward the debug impl without a surrounding "V3(...)", for somewhat
        // selfish reasons (it makes the debug output valid JSON and Python for
        // many types, significantly lowering the barrier to some common tasks
        // during debugging)
        impl<$T: fmt::Debug> fmt::Debug for $Cn<$T> {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
            { fmt::Debug::fmt(&self.0, f) }
        }
    }
}

// ---------------------------------------------------------------------------
