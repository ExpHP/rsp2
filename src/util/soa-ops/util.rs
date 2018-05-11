
pub(crate) fn zip_eq<As, Bs>(a: As, b: Bs) -> ::std::iter::Zip<As::IntoIter, Bs::IntoIter>
where
    As: IntoIterator, As::IntoIter: ExactSizeIterator,
    Bs: IntoIterator, Bs::IntoIter: ExactSizeIterator,
{
    let (a, b) = (a.into_iter(), b.into_iter());
    assert_eq!(a.len(), b.len());
    a.zip(b)
}

#[cfg(test)]
pub(crate) use self::drop_pusher::DropPusher;
#[cfg(test)]
mod drop_pusher {
    use ::std::rc::Rc;
    use ::std::cell::RefCell;

    pub(crate) struct DropPusher<T: Copy>(Rc<RefCell<Vec<T>>>, T);

    impl<T: Copy + 'static> DropPusher<T> {
        /// Create a shared vector, and a `new` function which constructs
        /// `DropPushers` tied to that vector.
        pub fn new_trial() -> (Rc<RefCell<Vec<T>>>, Box<Fn(T) -> DropPusher<T>>)
        {
            let history = Rc::new(RefCell::new(vec![]));
            let new = {
                let history = history.clone();
                Box::new(move |x| DropPusher(history.clone(), x))
            };
            (history, new)
        }
    }

    #[cfg(test)]
    impl<T: Copy> Drop for DropPusher<T> {
        fn drop(&mut self) {
            self.0.borrow_mut().push(self.1);
        }
    }
}
