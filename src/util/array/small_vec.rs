use ::traits::IsArray;

use ::std::mem;
use ::std::ptr;

/// Construct an array from a function on indices.
///
/// `V` should be an array type, like `[T; n]`.
pub fn vec_from_fn<V, F>(f: F) -> V
  where
    V: ArrayFromFunctionExt,
    F: FnMut(usize) -> V::Element,
{ V::from_fn(f) }

/// Construct an array fallibly, short-circuiting on the first Error.
pub fn try_vec_from_fn<V, E, F>(f: F) -> Result<V, E>
  where
    V: ArrayFromFunctionExt,
    F: FnMut(usize) -> Result<V::Element, E>,
{ V::try_from_fn(f) }

/// Construct an array fallibly, short-circuiting on the first None.
///
/// (you hear the author mumble something incomprehensible about monads)
pub fn opt_vec_from_fn<V, F>(f: F) -> Option<V>
  where
    V: ArrayFromFunctionExt,  
    F: FnMut(usize) -> Option<V::Element>,
{ V::opt_from_fn(f) }

/// Implementation detail of `vec_from_fn` and `mat_from_fn`.
///
/// If you could just ignore this, that'd be swell.
/// Please prefer the freestanding functions instead.
pub trait ArrayFromFunctionExt: IsArray
{
    fn from_fn<F>(f: F) -> Self
    where F: FnMut(usize) -> Self::Element;

    fn try_from_fn<E, F>(f: F) -> Result<Self, E>
    where F: FnMut(usize) -> Result<Self::Element, E>;

    fn opt_from_fn<F>(f: F) -> Option<Self>
    where F: FnMut(usize) -> Option<Self::Element>;
}


impl<V: IsArray> ArrayFromFunctionExt for V
{
    fn from_fn<F>(mut f: F) -> Self
    where F: FnMut(usize) -> Self::Element
    {
        // SAFETY:
        //  - uninitialized() data must never be read; beware of drops!
        //  - ptr::write argument must be aligned
        //    - [T; n] is aligned to T
        //  - ptr::write leaks the old value
        let mut out = mem::ManuallyDrop::new(unsafe { mem::uninitialized::<Self>() });
        for i in 0..Self::array_len() {
            // If the function panics, uninit data remaining in `out` is safely
            // forgotten thanks to ManuallyDrop.  Any written values are leaked.
            unsafe { ptr::write(&mut out.array_as_mut_slice()[i], f(i)) };
        }
        mem::ManuallyDrop::into_inner(out)
    }

    fn try_from_fn<E, F>(mut f: F) -> Result<Self, E>
    where F: FnMut(usize) -> Result<Self::Element, E>
    {
        // SAFETY:
        //  - uninitialized() data must never be read; beware of drops!
        //  - ptr::{write, read} argument must be aligned
        //    - [T; n] is aligned to T
        //  - ptr::write leaks the old value
        //  - ptr::read creates the potential for double-drops
        let mut out = mem::ManuallyDrop::new(unsafe { mem::uninitialized::<Self>() });
        for i in 0..Self::array_len() {
            // If the function panics, uninit data remaining in `out` is safely
            // forgotten thanks to ManuallyDrop.  Any written values are leaked.
            match f(i) {
                Ok(x) => unsafe { ptr::write(&mut out.array_as_mut_slice()[i], x) },
                Err(e) => {
                    // Drop each element written. This drop could also panic;
                    // but the prior justifications for panic-safety still hold.
                    for p in out.array_as_slice()[..i].iter().rev() {
                        unsafe { ptr::read(p) }; // drop!
                    }
                    return Err(e);
                }
            }
        }
        Ok(mem::ManuallyDrop::into_inner(out))
    }

    fn opt_from_fn<F>(mut f: F) -> Option<Self>
    where F: FnMut(usize) -> Option<Self::Element>
    {
        // hand the problem off to our "sufficiently smart compiler"
        Self::try_from_fn(|i| f(i).map(Ok).unwrap_or(Err(()))).ok()
    }
}

/// Extension trait for folding arrays by value.
///
/// The `Copy` bound is a temporary(?) crutch for safety considerations;
/// it prevents usage with types that have a non-trivial `Drop`.
pub trait ArrayFoldExt: IsArray
  where Self::Element: Copy
{
    fn fold<B, F>(self, initial: B, mut f: F) -> B
    where F: FnMut(B, Self::Element) -> B {
        // SAFETY:
        //  - ptr::read argument must be aligned
        //  - ptr::read creates the potential for double-drops

        let array = mem::ManuallyDrop::new(self);
        let mut acc = initial;
        for p in array.array_as_slice() {
            // f() can panic.  If this happens, acc is dropped (not a problem),
            // and then the array is leaked using ManuallyDrop to avoid double-drops
            // of previously-read elements.
            acc = f(acc, unsafe { ptr::read(p) });
        }
        acc
    }
}

impl<V: IsArray> ArrayFoldExt for V
  where V::Element: Copy,
{ }

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    /// Pushes an element to a vector when dropped.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct PushDrop<'a, T: 'a>(Option<T>, &'a RefCell<Vec<T>>);
    impl<'a, T> PushDrop<'a, T> {
        fn new(x: T, cell: &'a RefCell<Vec<T>>) -> Self { PushDrop(Some(x), cell) }
    }

    impl<'a, T> Drop for PushDrop<'a, T> {
        fn drop(&mut self) {
            let x = self.0.take().unwrap();
            self.1.borrow_mut().push(x)
        }
    }

    #[test]
    fn try_vec_from_fn_drop() {
        use super::try_vec_from_fn;

        let vec = RefCell::new(vec![3, 4, 2]);

        // Completely construct something;
        // nothing should get dropped.
        let arr: Result<[PushDrop<i32>; 5], ()> = try_vec_from_fn(
            |i| Ok(PushDrop::new(i as i32, &vec))
        );
        assert_eq!(*vec.borrow(), vec![3, 4, 2]);
        ::std::mem::forget(arr);

        // Interrupt construction with an Err.
        // The successfully added elements should be dropped in reverse.
        let ret: Result<[PushDrop<i32>; 5], _> = try_vec_from_fn(
            |i| match i {
                3 => Err("lol!"),
                i => Ok(PushDrop::new(i as i32, &vec)),
            }
        );
        assert_eq!(ret, Err("lol!"));
        assert_eq!(*vec.borrow(), vec![3, 4, 2, 2, 1, 0]);
    }
}