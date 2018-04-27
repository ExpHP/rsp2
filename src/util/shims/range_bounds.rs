
pub use self::stuff_from_core::*;
mod stuff_from_core {
    // MOST OF THIS IS COPIED DIRECTLY FROM LIBCORE!!!
    //
    // Such code falls under the following terms:
    //
    //   Copyright 2012 The Rust Project Developers. See the COPYRIGHT
    //   file at the top-level directory of this distribution and at
    //   http://rust-lang.org/COPYRIGHT.
    //
    //   Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
    //   http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
    //   <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
    //   option. This file may not be copied, modified, or distributed
    //   except according to those terms.

    use ::std::collections::Bound;
    use ::std::ops::{Range, RangeFrom, RangeTo, RangeFull};
    #[cfg(not_yet)]
    use ::std::ops::{RangeToInclusive, RangeInclusive};

    /// `RangeBounds` is implemented by Rust's built-in range types, produced
    /// by range syntax like `..`, `a..`, `..b` or `c..d`.
    pub trait RangeBounds<T: ?Sized> {
        /// Start index bound.
        ///
        /// Returns the start value as a `Bound`.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(collections_range)]
        ///
        /// # fn main() {
        /// use std::ops::Bound::*;
        /// use std::ops::RangeBounds;
        ///
        /// assert_eq!((..10).start(), Unbounded);
        /// assert_eq!((3..10).start(), Included(&3));
        /// # }
        /// ```
        fn start(&self) -> Bound<&T>;

        /// End index bound.
        ///
        /// Returns the end value as a `Bound`.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(collections_range)]
        ///
        /// # fn main() {
        /// use std::ops::Bound::*;
        /// use std::ops::RangeBounds;
        ///
        /// assert_eq!((3..).end(), Unbounded);
        /// assert_eq!((3..10).end(), Excluded(&10));
        /// # }
        /// ```
        fn end(&self) -> Bound<&T>;
    }

    use self::Bound::{Excluded, Included, Unbounded};

    impl<T: ?Sized> RangeBounds<T> for RangeFull {
        fn start(&self) -> Bound<&T> {
            Unbounded
        }
        fn end(&self) -> Bound<&T> {
            Unbounded
        }
    }

    impl<T> RangeBounds<T> for RangeFrom<T> {
        fn start(&self) -> Bound<&T> {
            Included(&self.start)
        }
        fn end(&self) -> Bound<&T> {
            Unbounded
        }
    }

    impl<T> RangeBounds<T> for RangeTo<T> {
        fn start(&self) -> Bound<&T> {
            Unbounded
        }
        fn end(&self) -> Bound<&T> {
            Excluded(&self.end)
        }
    }

    impl<T> RangeBounds<T> for Range<T> {
        fn start(&self) -> Bound<&T> {
            Included(&self.start)
        }
        fn end(&self) -> Bound<&T> {
            Excluded(&self.end)
        }
    }

    #[cfg(not_yet)]
    impl<T> RangeBounds<T> for RangeInclusive<T> {
        fn start(&self) -> Bound<&T> {
            Included(&self.start)
        }
        fn end(&self) -> Bound<&T> {
            Included(&self.end)
        }
    }

    #[cfg(not_yet)]
    impl<T> RangeBounds<T> for RangeToInclusive<T> {
        fn start(&self) -> Bound<&T> {
            Unbounded
        }
        fn end(&self) -> Bound<&T> {
            Included(&self.end)
        }
    }

    impl<T> RangeBounds<T> for (Bound<T>, Bound<T>) {
        fn start(&self) -> Bound<&T> {
            match *self {
                (Included(ref start), _) => Included(start),
                (Excluded(ref start), _) => Excluded(start),
                (Unbounded, _)           => Unbounded,
            }
        }

        fn end(&self) -> Bound<&T> {
            match *self {
                (_, Included(ref end)) => Included(end),
                (_, Excluded(ref end)) => Excluded(end),
                (_, Unbounded)         => Unbounded,
            }
        }
    }

    impl<'a, T: ?Sized + 'a> RangeBounds<T> for (Bound<&'a T>, Bound<&'a T>) {
        fn start(&self) -> Bound<&T> {
            self.0
        }

        fn end(&self) -> Bound<&T> {
            self.1
        }
    }

    impl<'a, T> RangeBounds<T> for RangeFrom<&'a T> {
        fn start(&self) -> Bound<&T> {
            Included(self.start)
        }
        fn end(&self) -> Bound<&T> {
            Unbounded
        }
    }

    impl<'a, T> RangeBounds<T> for RangeTo<&'a T> {
        fn start(&self) -> Bound<&T> {
            Unbounded
        }
        fn end(&self) -> Bound<&T> {
            Excluded(self.end)
        }
    }

    impl<'a, T> RangeBounds<T> for Range<&'a T> {
        fn start(&self) -> Bound<&T> {
            Included(self.start)
        }
        fn end(&self) -> Bound<&T> {
            Excluded(self.end)
        }
    }

    #[cfg(not_yet)]
    impl<'a, T> RangeBounds<T> for RangeInclusive<&'a T> {
        fn start(&self) -> Bound<&T> {
            Included(self.start)
        }
        fn end(&self) -> Bound<&T> {
            Included(self.end)
        }
    }

    #[cfg(not_yet)]
    impl<'a, T> RangeBounds<T> for RangeToInclusive<&'a T> {
        fn start(&self) -> Bound<&T> {
            Unbounded
        }
        fn end(&self) -> Bound<&T> {
            Included(self.end)
        }
    }
}

use ::std::collections::Bound;
use ::std::ops::Range;

pub trait RangeBoundsHelper: RangeBounds<usize> {
    fn to_range(&self, len: usize) -> Range<usize>
    {
        let inclusive_start = match self.start() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1,
            Bound::Unbounded => 0,
        };

        let exclusive_end = match self.end() {
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => len,
        };

        inclusive_start..exclusive_end
    }
}

impl<T: RangeBounds<usize>> RangeBoundsHelper for T {}
