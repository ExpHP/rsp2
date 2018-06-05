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

use ::traits::IsArray;

use ::std::ptr;
use ::std::ops::Range;
use ::std::mem::ManuallyDrop;

pub trait ArrayMoveIterExt: IsArray {
    /// A true `into_iter()` for arrays.
    fn move_iter(self) -> MoveIter<Self>;
}

pub struct MoveIter<A: IsArray>{
    array: ManuallyDrop<A>,
    range: Range<usize>,
}

impl<A: IsArray> MoveIter<A> {
    #[inline]
    pub fn new(array: A) -> Self
    { MoveIter {
        array: ManuallyDrop::new(array),
        range: 0..A::array_len(),
    }}
}

impl<A: IsArray> Drop for MoveIter<A> {
    fn drop(&mut self)
    {
        for _ in self.by_ref() { }
        assert_eq!(self.len(), 0);
    }
}

impl<A: IsArray> Iterator for MoveIter<A> {
    type Item = A::Element;

    #[inline]
    fn next(&mut self) -> Option<A::Element>
    {
        self.range.next().map(|index| {
            unsafe { ptr::read(&self.array.array_as_slice()[index]) }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>)
    { (self.len(), Some(self.len())) }
}

impl<A: IsArray> ExactSizeIterator for MoveIter<A> {
    #[inline]
    fn len(&self) -> usize
    { self.range.len() }
}

impl<A: IsArray> DoubleEndedIterator for MoveIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A::Element>
    {
        self.range.next_back().map(|index| {
            unsafe { ptr::read(&self.array.array_as_slice()[index]) }
        })
    }
}

impl<A: IsArray> ::std::iter::FusedIterator for MoveIter<A> { }

macro_rules! impl_iter_move {
    ($n:expr) => {
        impl<T> ArrayMoveIterExt for [T; $n] {
            #[inline]
            fn move_iter(self) -> MoveIter<[T; $n]>
            { MoveIter::new(self) }
        }
    };
}

each_array_size!{ impl_iter_move!{0...32} }

#[cfg(test)]
#[deny(dead_code)]
mod tests {
    use ::test_util::PushDrop;

    use ::std::cell::RefCell;
    use super::*;

    // FIXME there ought to be tests of panic safety, using catch_unwind

    #[test]
    fn is_fused() {
        let mut it = [2, 3, 4].move_iter();
        assert_eq!(it.next_back(), Some(4));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);

        let mut it = [2, 3, 4].move_iter().rev();
        assert_eq!(it.next_back(), Some(2));
        assert_eq!(it.next(), Some(4));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }

    #[test]
    fn drop_behavior() {
        use ::arr_from_fn;
        let vec = RefCell::new(vec![]);

        let arr: [PushDrop<i32>; 6] = arr_from_fn(
            |i| PushDrop::new(i as i32, &vec)
        );
        assert_eq!(*vec.borrow(), vec![]);

        {
            let mut it = arr.move_iter();
            {
                let _elem_0 = it.next();
                {
                    let _elem_5 = it.next_back();

                    // nothing has been dropped yet
                    assert_eq!(*vec.borrow(), vec![]);
                    // on scope exit, drop the last element
                }
                assert_eq!(*vec.borrow(), vec![5]);
                // on scope exit, drop the first element
            }
            assert_eq!(*vec.borrow(), vec![5, 0]);
            // on scope exit, drop everything else
        }
        assert_eq!(*vec.borrow(), vec![5, 0, 1, 2, 3, 4]);
    }
}
