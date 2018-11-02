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

/// Simple utility type for tracking changes in data.
pub(crate) struct MaybeDirty<T> {
    // NOTE: Possible states for the members:
    //
    //        dirty:       clean:       when
    //        Some(s)       None       is dirty, and has never been clean.
    //        Some(s)      Some(s)     is dirty, but has been clean in the past.
    //         None        Some(s)     is currently clean.

    /// new data that has not been marked clean.
    dirty: Option<T>,
    /// the last data marked clean.
    clean: Option<T>,
}

impl<T> MaybeDirty<T> {
    pub fn new_dirty(x: T) -> MaybeDirty<T> {
        MaybeDirty {
            dirty: Some(x),
            clean: None,
        }
    }

    pub fn is_dirty(&self) -> bool
    { self.dirty.is_some() }

    pub fn last_clean(&self) -> Option<&T>
    { self.clean.as_ref() }

    pub fn get(&self) -> &T
    { self.dirty.as_ref().or(self.last_clean()).unwrap() }

    /// Get a mutable reference. This automatically marks the value as dirty.
    pub fn get_mut(&mut self) -> &mut T
        where T: Clone,
    {
        if self.dirty.is_none() {
            self.dirty = self.clean.clone();
        }
        self.dirty.as_mut().unwrap()
    }

    pub fn mark_clean(&mut self)
    {
        assert!(self.dirty.is_some() || self.clean.is_some());

        if self.dirty.is_some() {
            self.clean = self.dirty.take();
        }

        assert!(self.dirty.is_none());
        assert!(self.clean.is_some());
    }

    // test if f(x) is dirty by equality
    // HACK
    // this is only provided to help work around borrow checker issues
    //
    // To clarify: If there is no last clean value, then ALL projections
    // are considered dirty by definition.
    pub fn is_projection_dirty<K: ?Sized + PartialEq>(
        &self,
        mut f: impl FnMut(&T) -> &K,
    ) -> bool {
        match (&self.clean, &self.dirty) {
            (Some(clean), Some(dirty)) => f(clean) != f(dirty),
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => unreachable!(),
        }
    }

    // HACK
    // This differs from `is_projection_dirty` only in that the callback
    // returns owned data instead of borrowed. One might think that this
    // method could therefore be used to implement the other; but it can't,
    // because the lifetime in F's return type would be overconstrained.
    pub fn is_function_dirty<K: PartialEq>(
        &self,
        mut f: impl FnMut(&T) -> K,
    ) -> bool {
        match (&self.clean, &self.dirty) {
            (Some(clean), Some(dirty)) => f(clean) != f(dirty),
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => unreachable!(),
        }
    }
}
