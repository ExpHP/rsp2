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

use ::{Perm, Permute};
use ::std::iter::FusedIterator;
use ::frunk::{HNil, HCons};

/// Type of "a thing that has been partitioned."
///
/// It is a `V` that has been broken up into many smaller `V`s,
/// each associated with a label `L`. (such as a layer number)
///
/// Typically, these labels are unique. However, technically there is
/// nothing stopping you from constructing one with duplicate labels.
/// (if you do, they will still be considered to represent separate pieces)
pub type Parted<L, V> = Vec<(L, V)>;

/// A partition operator.
///
/// It represents a specific way to break up a vector into smaller vectors,
/// each with a label of type `L`. These labels are *usually* (but not
/// necessarily) unique, and the partitions are not necessarily contiguous
/// in the original vector. Basically, `Part` contains a partitioned form
/// of the vector's indices.
///
/// Using the [`Partition`] trait, a `Part<L>` can be applied to any
/// `V` that implements `Partition` in order to break it up into a
/// [`Parted<L, V>`].
///
/// [`Partition`]: ../trait.Partition.html
/// [`Parted`]: ../type.Parted.html
#[derive(Debug, Clone)]
pub struct Part<L> {
    // the total size, precomputed.
    // (in the future, incomplete partitions may be allowed, in which
    //  case storing this will be not just a convenience, but manditory)
    index_limit: usize,
    part: Parted<L, Vec<usize>>,
}

// NOTE: Possible future extension:
//
// pub trait Backing<L>
//   : FromIterator<(L, Vec<usize>)>
//   + IntoIterator<Item=(L, Vec<usize>)>
// { }
//
// struct Part<L, P> { index_limit: usize, part: P }
// where L: Label<P>, P: Backing<L>;
//
// impl<L: Ord> Label<BTreeMap<L, Vec<usize>> for L;
// impl<L: Hash + Eq> Label<HashMap<L, Vec<usize>> for L;
// impl<L> Label<Vec<(L, Vec<usize>)>> for L;

#[derive(Debug, Fail)]
#[fail(display = "Tried to construct an invalid partition.")]
pub struct InvalidPartitionError(::failure::Backtrace);

impl<L> Part<L> {
    /// Create a partition that decomposes a vector entirely.
    ///
    /// Every integer from 0 to the total length must appear once.
    pub fn new(part: Parted<L, Vec<usize>>) -> Result<Self, InvalidPartitionError>
    {Ok({
        let index_limit = part.iter().map(|(_, v)| v.len()).sum();
        if !Self::validate_part(&part, index_limit) {
            return Err(InvalidPartitionError(::failure::Backtrace::new()))
        }
        Self { part, index_limit }
    })}

    /// Iterate over the keys for each region.
    ///
    /// Item type is `&L`;
    pub fn region_keys(&self) -> impl VeclikeIterator<Item=&L>
    { self.part.iter().map(|(label, _)| label) }

    /// Iterate over the index vectors for each region.
    ///
    /// Item type is `&[usize]`;
    pub fn region_indices(&self) -> impl VeclikeIterator<Item=&[usize]>
    { self.part.iter().map(|(_, idx)| &idx[..]) }

    /// # Safety
    ///
    /// Usage of the constructed `Part<L>` may lead to Undefined Behavior
    /// if either of the following conditions are violated:
    /// - no usize index may appear more than once in the input.
    /// - `index_limit` must equal the total size.
    pub unsafe fn new_unchecked(part: Parted<L, Vec<usize>>, index_limit: usize) -> Self
    { Part { part, index_limit } }

    /// Construct a Part from a sequence of keys, with one
    /// partition for each distinct value in the sequence.
    ///
    /// The partitions in the output will be sorted by key.
    pub fn from_ord_keys<Ls>(labels: Ls) -> Self
    where
        L: Ord,
        Ls: IntoIterator<Item=L>,
    {
        let mut map = ::std::collections::BTreeMap::new();
        for (i, key) in labels.into_iter().enumerate() {
            map.entry(key)
                .or_insert_with(Vec::new)
                .push(i);
        }
        Self::new(map.into_iter().collect()).expect("bug!")
    }

    pub fn into_parted_indices(self) -> Vec<(L, Vec<usize>)>
    { self.part }

    pub fn key_vec(&self) -> Vec<&L>
    {
        use ::std::mem;

        let mut temp = vec![None; self.index_limit];
        for (label, indices) in &self.part {
            for &i in indices {
                temp[i] = Some(label);
            }
        }

        // Take advantage of rust's null pointer optimization;
        debug_assert!(temp.iter().all(|x| x.is_some()));
        assert_eq!(mem::size_of::<&L>(), mem::size_of::<Option<&L>>());
        assert_eq!(mem::align_of::<&L>(), mem::align_of::<Option<&L>>());
        unsafe { mem::transmute::<Vec<Option<&L>>, Vec<&L>>(temp) }
    }

    pub fn owned_key_vec(&self) -> Vec<L>
    where L: Clone,
    { self.key_vec().into_iter().cloned().collect() }

    /// Get the permutation that would restore the original order of the data,
    /// if one were to simply concatenate the partitioned structures produced
    /// by applying this `Part`.
    pub fn restoring_perm(&self) -> Perm
    {
        // get the composite perm that produces the concatenated order from the original
        // (note: not the same as `composite_perm_for_part_lifo` because concatenation
        //        produces the fifo order)
        let mut perm_vec = vec![];
        for (_, ref piece) in &self.part {
            // (due to the convention that the order within a part matches the original order,
            //  we must sort the indices in each part)
            // (FIXME: maybe they should be stored as already sorted in the Part, since
            //         the order of indices in Part's partitions never matters anyways?)
            let mut sorted_piece = piece.clone();
            sorted_piece.sort();
            perm_vec.extend(sorted_piece);
        }

        // invert to restore the original order
        Perm::from_raw_inv(perm_vec).expect("BUG")
    }

    fn validate_part(part: &Parted<L, Vec<usize>>, index_limit: usize) -> bool
    {
        let slices: Vec<_> = part.iter().map(|(_, v)| &v[..]).collect();
        let mut xs = slices.concat();
        xs.sort();
        xs.into_iter().eq(0..index_limit)
    }
}

impl<L> IntoIterator for Part<L> {
    type Item = (L, Vec<usize>);
    type IntoIter = ::std::vec::IntoIter<(L, Vec<usize>)>;
    fn into_iter(self) -> Self::IntoIter
    { self.part.into_iter() }
}

/// Return type of `into_unlabeled_partitions`.  You hafta use it, sorry.
///
/// One day, we will be able to replace it with an associated type on `Partition`
/// (once existential types are supported on traits).
pub type Unlabeled<'a, T> = Box<VeclikeIterator<Item=T> + 'a>;

pub trait VeclikeIterator: ExactSizeIterator + DoubleEndedIterator + FusedIterator {}

impl<T> VeclikeIterator for T
where T: ExactSizeIterator + DoubleEndedIterator + FusedIterator {}

/// Trait for applying a `Part` to a `Vec` (or similar type), breaking it into pieces.
///
/// By making this a trait, it can be implemented on types like rsp2's own
/// `Coords` or anything else that contains data per-atom (such as eigenvectors).
///
/// The lifetime argument ensures that the iterator returned by `into_unlabeled_partitions`
/// does not outlive either Self or the partition; this allows the iterator to capture self
/// by value, and the partition by reference. Ultimately, we would need both GATs and
/// impl-Trait-on-Trait-impls to get rid of it.
pub trait Partition<'iter>: Sized + 'iter {
    /// Variant of `into_partitions` which composes more easily, and is
    /// therefore the one you need to implement.
    ///
    /// It returns an iterator over the partitions of `self`.
    ///
    /// See `into_partitions` for more info.
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>;

    /// Consume self to produce partitions.
    ///
    /// The ordering within each partition is specified, in order to allow
    /// composite types to reliably be able to defer to the implementations
    /// defined on each of their members.  This is not usually a concern
    /// since virtually all implementations will ultimately defer to `Vec<_>`
    /// for their implementation... but in case you must know:
    ///
    /// The ordering within each partition of the output must reflect the
    /// original order of those elements relative to each other in the
    /// input vec, rather than the order of the indices in `part`.
    fn into_partitions<L: Clone>(self, part: &'iter Part<L>) -> Parted<L, Self>
    { ::util::zip_eq(part.region_keys().cloned(), self.into_unlabeled_partitions(part)).collect() }
}

impl<'iter, T: 'iter> Partition<'iter> for Vec<T> {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {
        // permute all data for the first group to the very end,
        // with the second group before it, and etc.
        let mut data = self.permuted_by(&composite_perm_for_part_lifo(part));

        Box::new({
             part.region_indices().map(|x| x.len())
                 .map(move |region_len| {
                     let start = data.len() - region_len;
                     data.drain(start..).collect()
                 })
        })
    }
}

/// Helper function to generate a `Perm` that is useful for `Partition` impls:
///
/// The `Perm` returned permutes a vector so that all data for the first label
/// comes last, with all the data for the second label before it, and so on.
/// Ordering within each label follows the order required by the `Partition` trait.
pub fn composite_perm_for_part_lifo<L>(part: &Part<L>) -> ::Perm
{
    let mut sort_keys = vec![::std::usize::MAX; part.index_limit];

    // rev() so that the last region gets a key of 0
    for (int_key, indices) in part.region_indices().into_iter().rev().enumerate() {
        for &i in indices {
            sort_keys[i] = int_key;
        }
    }
    debug_assert!(sort_keys.iter().all(|&x| x != ::std::usize::MAX));
    Perm::argsort(&sort_keys)
}

impl<'iter, A, B> Partition<'iter> for (A, B)
where
    A: Partition<'iter> + 'iter,
    B: Partition<'iter> + 'iter,
{
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {
        let a_parted = self.0.into_unlabeled_partitions(part);
        let b_parted = self.1.into_unlabeled_partitions(part);
        Box::new(::util::zip_eq(a_parted, b_parted))
    }
}

impl<'iter> Partition<'iter> for HNil {
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {Box::new({
        part.region_keys().map(|_| HNil)
    })}
}

impl<'iter, A, B> Partition<'iter> for HCons<A, B>
where
    A: Partition<'iter> + 'iter,
    B: Partition<'iter> + 'iter,
{
    fn into_unlabeled_partitions<L>(self, part: &'iter Part<L>) -> Unlabeled<'iter, Self>
    {Box::new({
        self.pop().into_unlabeled_partitions(part)
            .map(|(head, tail)| HCons { head, tail })
    })}
}


#[cfg(test)]
#[deny(dead_code)]
mod tests {
    use super::*;
    use ::rand::Rng;

    // FIXME: I don't see any tests where two labels have the same value...

    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
    pub enum LetterKind { Vowel, Consonant }
    impl LetterKind {
        fn of(c: char) -> Self
        { match c {
            'a' | 'e' | 'i' | 'o' | 'u' => LetterKind::Vowel,
            _ => LetterKind::Consonant,
        }}
    }

    #[test]
    fn from_ord_labels() {
        let kinds =
            vec!['a', 'b', 'c', 'd', 'e', 'f'].into_iter()
            .map(LetterKind::of).collect::<Vec<_>>();

        assert_eq!(
            Part::from_ord_keys(kinds).into_parted_indices(),
            vec![
                (LetterKind::Vowel, vec![0, 4]),
                (LetterKind::Consonant, vec![1, 2, 3, 5]),
            ],
        );
    }

    #[test]
    fn basic() {
        let vec = vec!['a', 'b', 'c', 'd', 'e', 'f'];
        let part = vec![
            (LetterKind::Vowel, vec![0, 4]),
            // (also test indices out of sorted order)
            (LetterKind::Consonant, vec![5, 1, 2, 3]),
        ];
        let part = Part::new(part).unwrap();

        let actual = vec.into_partitions(&part);
        assert_eq!(
            actual,
            vec![
                (LetterKind::Vowel, vec!['a', 'e']),
                // specified order is "the order within the original vec"
                (LetterKind::Consonant, vec!['b', 'c', 'd', 'f']),
            ],
        );
    }

    #[test]
    fn empty() {
        let part: Vec<((), Vec<usize>)> = vec![];
        let part = Part::new(part).unwrap();

        assert_eq!(
            Vec::<()>::new().into_partitions(&part),
            vec![],
        );

        // an empty partition in a non-empty Part
        let vec = vec!['a', 'b', 'c', 'd', 'e', 'f'];
        let part = Part::new(vec![
            (0, vec![1, 5]),
            (1, vec![]),
            (2, vec![4, 0, 2, 3]),
        ]).unwrap();
        let expected = vec![
            (0, vec!['b', 'f']),
            (1, vec![]),
            (2, vec!['a', 'c', 'd', 'e']),
        ];
        assert_eq!(vec.into_partitions(&part), expected);
    }

    #[test]
    fn error() {
        // skipped an index
        assert_matches!{
            Err(InvalidPartitionError(..)),
            Part::new(vec![ ((), vec![0, 1, 2, 4]) ]),
        }

        // duplicate, same vec
        assert_matches!{
            Err(InvalidPartitionError(..)),
            Part::new(vec![ ((), vec![2, 1, 2, 4]) ]),
        }

        // duplicate, different vec
        assert_matches!{
            Err(InvalidPartitionError(..)),
            Part::new(vec![
                (true, vec![0, 2, 4]),
                (false, vec![1, 2, 5]),
            ]),
        }
    }

    #[test]
    fn drop() {
        let (drop_history, dp) = ::util::DropPusher::new_trial();

        {
            let vec = vec![dp('a'), dp('b'), dp('c'), dp('d'), dp('e'), dp('f')];

            let mut vec2 = vec.into_partitions(&Part::new(vec![
                (true, vec![3, 0, 1]),
                (false, vec![2, 4, 5]),
            ]).unwrap());
            assert_eq!(drop_history.borrow().len(), 0);

            let _ = vec2.pop();
            drop_history.borrow_mut().sort();
            assert_eq!(&*drop_history.borrow(), &vec!['c', 'e', 'f']);

            let _ = vec2.pop();
            drop_history.borrow_mut().sort();
            assert_eq!(&*drop_history.borrow(), &vec!['a', 'b', 'c', 'd', 'e', 'f']);
        }
        assert_eq!(&*drop_history.borrow(), &vec!['a', 'b', 'c', 'd', 'e', 'f']);
    }

    fn random_part(len: usize) -> Part<usize> {
        let mut remaining: Vec<_> = Perm::random(len).into_vec();
        let mut part = vec![];
        while !remaining.is_empty() {
            let label = part.len();

            let n = ::rand::thread_rng().gen_range(0, usize::min(4, remaining.len() + 1));
            let start = remaining.len() - n;
            part.push((label, remaining.drain(start..).collect()))
        }
        Part::new(part).unwrap()
    }

    #[test]
    fn restoring_perm() {
        assert_eq!(
            Part::<()>::new(vec![]).unwrap().restoring_perm(),
            Perm::eye(0),
        );

        let original: Vec<_> = "abcdefghijklmnop".chars().collect();
        for _ in 0..4 {
            let part = random_part(original.len());

            let restored = {
                original.clone()
                    .into_unlabeled_partitions(&part)
                    .collect::<Vec<_>>()
                    .concat()
                    .permuted_by(&part.restoring_perm())
            };
            assert_eq!(original, restored, "{:?}", part);
        }
    }
}
