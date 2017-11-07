use ::std::cell::RefCell;
use ::std::fmt::Debug;

/// Pushes an element to a vector when dropped.
///
/// This is useful for testing code that uses things
/// like `std::mem::ManuallyDrop` and `std::ptr::read`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PushDrop<'a, T: Debug + 'a>(Option<T>, &'a RefCell<Vec<T>>);
impl<'a, T: Debug> PushDrop<'a, T> {
    pub(crate) fn new(x: T, cell: &'a RefCell<Vec<T>>) -> Self { PushDrop(Some(x), cell) }
}

impl<'a, T: Debug> PushDrop<'a, T> {
    /// Destroy the PushDrop without consequence, producing the value
    /// that would have been pushed.
    pub(crate) fn into_inner(mut self) -> T
    {
        let x = self.0.take().unwrap();
        ::std::mem::forget(self);
        x
    }
}

impl<'a, T: Debug> Drop for PushDrop<'a, T> {
    fn drop(&mut self) {
        match self.0.take() {
            Some(x) => self.1.borrow_mut().push(x),
            None => panic!("Double-drop of a PushDrop detected!  Vector: {:?}",
                self.1.borrow())
        }
    }
}
