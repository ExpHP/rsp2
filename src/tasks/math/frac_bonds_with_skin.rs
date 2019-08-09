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

use crate::FailResult;
use rsp2_array_types::V3;
use rsp2_structure::bonds::FracBonds;
use rsp2_structure::{Coords, Lattice};

/// Simple wrapper around FracBonds that caches using a skin distance.
///
/// The potential must be such that distances beyond `meta_range(type_a, type_b)` produce
/// a potential and force of zero.  The skin distance then represents an additional amount
/// added to the FracBonds search radius to avoid needless recomputation.
pub struct FracBondsWithSkin<M, F: ?Sized> {
    last: Option<Cache<M>>,
    skin_distance: f64,

    // FIXME: This should just be F, but until `Box<Fn()>` is finally able to implement `Fn()`,
    //        the only way we can let the caller use dynamic polymorphism without introducing
    //        a lifetime bound is by forcing all callers to use `Box`.
    // NOTE: (the reason we need to support dynamic polymorphism is because that is currently
    //        the only substitute for existential types outside of function return types.)
    meta_range: Box<F>,

    // HACK
    num_calls: u64,
    check_frequency: u64,
}

struct Cache<M> {
    input_lattice: Lattice,
    input_carts: Vec<V3>,
    input_meta: Vec<M>,
    output: FracBonds,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum CacheStatus { Applicable, Invalidated }

impl<M, F: ?Sized> FracBondsWithSkin<M, F>
where
    F: Fn(&M, &M) -> Option<f64>,
    M: Ord,
{
    pub fn new(meta_range: Box<F>, skin_distance: f64) -> Self {
        let last = None;
        let num_calls = 0;
        let check_frequency = 1;
        Self { last, meta_range, skin_distance, num_calls, check_frequency }
    }

    pub fn set_check_frequency(&mut self, check_frequency: u64) {
        self.check_frequency = check_frequency;
    }

    /// Compute the `FracBonds` for a given structure, possibly reusing results cached from a
    /// previous call to `compute` to elide an expensive search.
    ///
    /// The output is guaranteed to include every bond within the `max_range` and `meta_range`
    /// parameters provided on construction.  It may also incidentally include some bonds of longer
    /// lengths (by up to about `2.0 * skin_depth`), but not necessarily all of them.
    pub fn compute(
        &mut self,
        coords: &Coords,
        meta: impl IntoExactSizeCloneIterator<Item=M>,
    ) -> FailResult<&FracBonds> {
        let meta = meta.into_exact_size_clone_iterator();

        let replace_cache = {
            match (self.num_calls, self.check_frequency) {
                (0, _) => true, // first call
                (_, 0) => false, // infinite delay
                (a, b) if a % b == 0 => {
                    match self.cache_status(coords, meta.clone()) {
                        CacheStatus::Invalidated => true,
                        CacheStatus::Applicable => false,
                    }
                },
                (_, _) => false,
            }
        };
        self.num_calls += 1;

        if replace_cache {
            self.last = Some(Cache {
                input_lattice: coords.lattice().clone(),
                input_carts: coords.to_carts(),
                input_meta: meta.clone().collect(),
                output: self.force_compute(coords, meta)?,
            });
        }
        Ok(&self.last.as_ref().unwrap().output)
    }

    fn cache_status(
        &self,
        coords: &Coords,
        meta: impl Clone + ExactSizeIterator<Item=M>,
    ) -> CacheStatus {
        assert_eq!(coords.len(), meta.len());

        let last = match self.last {
            None => return CacheStatus::Invalidated,
            Some(ref x) => x,
        };

        if &last.input_lattice != coords.lattice() {
            return CacheStatus::Invalidated;
        }

        if last.input_carts.len() != coords.len() {
            return CacheStatus::Invalidated;
        }

        if zip_eq!(&last.input_meta, meta).any(|(a, ref b)| a != b) {
            return CacheStatus::Invalidated;
        }

        // We consider necessary conditions for the cached FracBonds to be missing a bond:
        // - Two atoms initially beyond the search radius must now be within the interaction radius.
        // - This means they must have approached each other by at least `skin distance`.
        // - ...which means that at least one of them must have moved by `0.5 * skin_distance`.
        //
        // We fudge this amount down *just a tiny bit further* to be safe.
        let square = |x| x*x;
        let max_valid_sqnorm = square(0.5 * self.skin_distance) * (1.0 - 1e-9);
        for (&old, &new) in zip_eq!(&last.input_carts, &coords.to_carts()) {
            if (new - old).sqnorm() > max_valid_sqnorm {
                return CacheStatus::Invalidated;
            }
        }

        CacheStatus::Applicable
    }

    fn force_compute(
        &self,
        coords: &Coords,
        meta: impl Clone + ExactSizeIterator<Item=M>,
    ) -> FailResult<FracBonds> {
        FracBonds::compute_with_meta(
            coords, meta.clone(),
            |a, b| self.meta_search_range(a, b),
        )
    }

    fn meta_search_range(&self, a: &M, b: &M) -> Option<f64> {
        (self.meta_range)(a, b).map(|x| x + self.skin_distance)
    }
}

// FIXME:
// Workaround for there currently being no way to write
// `-> impl IntoIterator<IntoIter=impl Clone + ExactSizeIterator>`
pub trait IntoExactSizeCloneIterator {
    type Item;
    type IntoIter: Clone + ExactSizeIterator<Item=Self::Item>;

    fn into_exact_size_clone_iterator(self) -> Self::IntoIter;
}

impl<I> IntoExactSizeCloneIterator for I
where
    I: IntoIterator,
    <I as IntoIterator>::IntoIter: Clone + ExactSizeIterator<Item = <I as IntoIterator>::Item>,
{
    type Item = <I as IntoIterator>::Item;
    type IntoIter = <I as IntoIterator>::IntoIter;

    fn into_exact_size_clone_iterator(self) -> Self::IntoIter { self.into_iter() }
}
