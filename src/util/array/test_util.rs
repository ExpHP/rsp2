use std::cell::RefCell;

/// Pushes an element to a vector when dropped.
///
/// This is useful for testing code that uses things
/// like `std::mem::ManuallyDrop` and `std::ptr::read`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PushDrop<'a, T: 'a>(Option<T>, &'a RefCell<Vec<T>>);
impl<'a, T> PushDrop<'a, T> {
    pub(crate) fn new(x: T, cell: &'a RefCell<Vec<T>>) -> Self { PushDrop(Some(x), cell) }
}

impl<'a, T> Drop for PushDrop<'a, T> {
    fn drop(&mut self) {
        let x = self.0.take().unwrap();
        self.1.borrow_mut().push(x)
    }
}
