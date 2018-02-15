
/// Trait for wrapping newtypes that might be behind pointers.
///
/// This makes it easier to forward the implementation of a trait
/// method that takes `&self`.
///
/// # Safety
/// This trait has default method implementations which unsafely
/// cast pointers such as `Type<&T>` to `&Type<T>`.
///
/// Implementing this for anything but a newtype (`type Newtype<T>(pub T);`)
/// is **strongly inadvisable.**  The following are the *known* possible
/// sources of danger:
///
/// * It may be possible to synthesize invalid lifetimes or types.
/// * Behavior is undefined if the size (or worse, the sizedness)
///   of `T` and `Self` do not match.
pub unsafe trait IsNewtype<T: ?Sized> {
    /// The inverse of `&self.0` for a newtype.
    fn wrap_ref(x: &T) -> &Self
    { unsafe { transmute_copy_ref(x) } }

    /// The inverse of `&mut self.0` for a newtype.
    fn wrap_mut(x: &mut T) -> &mut Self
    { unsafe { transmute_copy_mut(x) } }

    /// The inverse of `Box::new(self.0)` for a newtype.
    fn wrap_box(x: Box<T>) -> Box<Self>
    { unsafe { transmute_copy_box(x) } }

    /// Basically just `Box::new(self.0)`.
    fn unwrap_box(x: Box<Self>) -> Box<T>
    { unsafe { transmute_copy_box(x) } }
}

/// `transmute_copy` restricted to references.
///
/// Behavior is undefined if A and B differ in sizedness.
/// (and of course, plenty of other undefined behavior is possible
///  depending on what type B is...)
#[allow(dead_code)]
unsafe fn transmute_copy_ref<A: ?Sized, B: ?Sized>(borrow: &A) -> &B
{
    #![allow(unused_unsafe)]
    let ptr = borrow as *const A;
    // we can't just cast ptr to *const B because they may differ in fatness
    let ptrptr = (&ptr) as *const *const A as *const *const B;
    let ptr = unsafe { *ptrptr };
    unsafe { ptr.as_ref().unwrap() }
}

/// `transmute_copy` restricted to mutable references.
///
/// Behavior is undefined if A and B differ in sizedness.
/// (and of course, plenty of other undefined behavior is possible
///  depending on what type B is...)
#[allow(dead_code)]
unsafe fn transmute_copy_mut<A: ?Sized, B: ?Sized>(borrow: &mut A) -> &mut B
{
    #![allow(unused_unsafe)]
    let ptr = borrow as *mut A;
    // we can't just cast ptr to *mut B because they may differ in fatness
    let ptrptr = (&ptr) as *const *mut A as *const *mut B;
    let ptr = unsafe { *ptrptr };
    unsafe { ptr.as_mut().unwrap() }
}

/// `transmute_copy` restricted to references.
///
/// Behavior is undefined if A and B differ in sizedness.
/// (and of course, plenty of other undefined behavior is possible
///  depending on what type B is...)
unsafe fn transmute_copy_box<A: ?Sized, B: ?Sized>(x: Box<A>) -> Box<B>
{
    #![allow(unused_unsafe)]
    let ptr = Box::into_raw(x);
    // we can't just cast ptr to *const B because they may differ in fatness
    let ptrptr = (&ptr) as *const *mut A as *const *mut B;
    let ptr = unsafe { *ptrptr };
    unsafe { Box::from_raw(ptr) }
}
