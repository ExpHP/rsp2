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

use std::mem;

use super::{V2, V3, V4};

/// Zero-cost transformations from sequences of arrays into sequences of `Vn`.
///
/// # Safety
///
/// The default impls effectively perform `transmute`, and some of the generic
/// impls assume that it is safe to perform pointer casts between Self and `Self::En`.
/// (this may be done even on pointers to pointers, or smart pointers and etc.)
pub unsafe trait Envee {
    type En: ?Sized;

    /// Casts a sequence of arrays into `V2`/`V3`/`V4`s.
    #[inline(always)]
    fn envee(self) -> Self::En
    where Self: Sized, Self::En: Sized
    { unsafe { mem::transmute_copy(&mem::ManuallyDrop::new(self)) } }

    /// Borrow a sequence of arrays as `V2`/`V3`/`V4`s.
    ///
    /// This method exists for the convenience of autoref. (Contrast with `(&self).envee()`)
    #[inline(always)]
    fn envee_ref(&self) -> &Self::En { self.envee() }

    /// Mutably borrow a sequence of arrays as `V2`/`V3`/`V4`s.
    ///
    /// This method exists for the convenience of autoref. (Contrast with `(&mut self).envee()`)
    #[inline(always)]
    fn envee_mut(&mut self) -> &mut Self::En { self.envee() }
}

/// Zero-cost transformations from sequences of `Vn` into sequences of arrays.
///
/// # Safety
///
/// The default impls effectively perform `transmute`, and some of the generic
/// impls assume that it is safe to perform pointer casts between Self and `Self::Un`.
/// (this may be done even on pointers to pointers, or smart pointers and etc.)
pub unsafe trait Unvee {
    type Un: ?Sized;

    /// Casts a sequence of `V2`/`V3`/`V4`s into arrays.
    #[inline(always)]
    fn unvee(self) -> Self::Un
    where Self: Sized, Self::Un: Sized
    { unsafe { mem::transmute_copy(&mem::ManuallyDrop::new(self)) } }

    /// Borrow a sequence of `V2`/`V3`/`V4`s as arrays.
    ///
    /// This method exists for the convenience of autoref. (Contrast with `(&self).unvee()`)
    #[inline(always)]
    fn unvee_ref(&self) -> &Self::Un { self.unvee() }

    /// Mutably borrow a sequence of `V2`/`V3`/`V4`s as arrays.
    ///
    /// This method exists for the convenience of autoref. (Contrast with `(&mut self).unvee()`)
    #[inline(always)]
    fn unvee_mut(&mut self) -> &mut Self::Un { self.unvee() }
}

gen_each!{
    @{Vn_n}
    for_each!( {$Vn:ident $n:tt} ) => {
        unsafe impl<X> Envee for [[X;$n]] { type En = [$Vn<X>]; }
        unsafe impl<X> Unvee for [$Vn<X>] { type Un = [[X;$n]]; }

        unsafe impl<X> Envee for Vec<[X;$n]> { type En = Vec<$Vn<X>>; }
        unsafe impl<X> Unvee for Vec<$Vn<X>> { type Un = Vec<[X;$n]>; }
    }
}

gen_each!{
    @{Vn_n}
    @{0...8}
    for_each!( {$Vn:ident $n:tt} {$k:tt} ) => {
        unsafe impl<X> Envee for [[X;$n]; $k] { type En = [$Vn<X>; $k]; }
        unsafe impl<X> Unvee for [$Vn<X>; $k] { type Un = [[X;$n]; $k]; }
    }
}

mod envee_generic_impls {
    use super::*;

    use ::std::rc::{Rc, Weak as RcWeak};
    use ::std::sync::{Arc, Weak as ArcWeak};
    use ::std::cell::RefCell;

    gen_each!{
        [ {Envee En} {Unvee Un} ]
        for_each!( {$Envee:ident $En:ident} ) => {
            unsafe impl<'a, V: $Envee + ?Sized> $Envee for &'a V      { type $En = &'a V::$En; }
            unsafe impl<'a, V: $Envee + ?Sized> $Envee for &'a mut V  { type $En = &'a mut V::$En; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for Box<V>     { type $En = Box<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for Rc<V>      { type $En = Rc<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for RcWeak<V>  { type $En = RcWeak<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for Arc<V>     { type $En = Arc<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for ArcWeak<V> { type $En = ArcWeak<V::$En>; }
            unsafe impl<    V: $Envee + ?Sized> $Envee for RefCell<V> { type $En = RefCell<V::$En>; }
        }
    }
}
