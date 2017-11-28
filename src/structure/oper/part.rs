use ::{Result, ErrorKind};
use ::oper::Permute;

pub type Parted<L, V> = Vec<(L, V)>;
#[derive(Debug, Clone)]
pub struct Part<L> {
    // the total size, precomputed.
    // (in the future, incomplete partitions may be allowed, in which
    //  case storing this will be just a convenience, but manditory)
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

impl<L> Part<L> {
    /// Create a partition that decomposes a vector entirely.
    ///
    /// Every integer from 0 to the total length must appear once.
    fn new(part: Parted<L, Vec<usize>>) -> Result<Self>
    {Ok({
        let index_limit = part.iter().map(|&(_, ref v)| v.len()).sum();
        ensure!(Self::validate_part(&part, index_limit), ErrorKind::BadPart);
        Self { part, index_limit }
    })}

    fn validate_part(part: &Parted<L, Vec<usize>>, index_limit: usize) -> bool
    {
        let slices: Vec<_> = part.iter().map(|&(_, ref v)| &v[..]).collect();
        let mut xs = slices.concat();
        xs.sort();
        xs.into_iter().eq(0..index_limit)
    }

    /// # Safety
    ///
    /// Usage of the constructed `Part<L>` may lead to Undefined Behavior
    /// if either of the following conditions are violated:
    /// - no usize index may appear more than once in the input.
    /// - `index_limit` must equal the total size.
    unsafe fn new_unchecked(part: Parted<L, Vec<usize>>, index_limit: usize) -> Self
    { Part { part, index_limit } }
}

impl<L> IntoIterator for Part<L> {
    type Item = (L, Vec<usize>);
    type IntoIter = ::std::vec::IntoIter<(L, Vec<usize>)>;
    fn into_iter(self) -> Self::IntoIter
    { self.part.into_iter() }
}

pub trait Partition: Sized {
    // NOTE: Specifying order allows composite types
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
    fn into_partitions<L: Clone>(self, part: &Part<L>) -> Parted<L, Self>;
}

impl<T> Partition for Vec<T> {
    fn into_partitions<L: Clone>(self, part: &Part<L>) -> Parted<L, Self>
    {
        // NOTE: this did not need unsafe code, but we reserve the right
        //       to make it unsafe in the future.

        let (labels, index_vecs): (Vec<&L>, Vec<_>) =
            part.part.iter()
                .map(|&(ref lbl, ref idxs)| (lbl, idxs.clone()))
                .unzip();

        let label_sizes: Vec<_> = index_vecs.iter().map(|v| v.len()).collect();

        // Permute so that all data for the last label comes first,
        // followed by all the data for the second to last label,
        // and so on.
        let perm = {
            let mut sort_keys = vec![::std::usize::MAX; part.index_limit];

            // the rev() inverts the values assigned by enumerate()
            // so that last group has the lowest key
            for (key, indices) in index_vecs.into_iter().rev().enumerate() {
                for i in indices {
                    sort_keys[i] = key;
                }
            }
            debug_assert!(sort_keys.iter().all(|&x| x != ::std::usize::MAX));
            ::oper::perm::argsort(&sort_keys)
        };

        let mut data = self.permuted_by(&perm);

        // read the first label's group (at the end), then the second...
        let mut out = Vec::with_capacity(labels.len());
        for (label, size) in ::util::zip_eq(labels, label_sizes) {
            let start = data.len() - size;
            out.push((label.clone(), data.drain(start..).collect()))
        }

        out
    }
}

impl<A, B> Partition for (A, B)
where
    A: Partition,
    B: Partition,
{
    fn into_partitions<L: Clone>(self, part: &Part<L>) -> Parted<L, (A, B)>
    {
        let a_parted = self.0.into_partitions(part);
        let b_parted = self.1.into_partitions(part);
        ::util::zip_eq(a_parted, b_parted)
            .map(|((label, a), (_, b))| (label, (a, b)))
            .collect()
    }
}

#[cfg(test)]
#[deny(dead_code)]
mod tests {
    use super::*;
    use ::Error;

    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
    pub enum LetterKind { Vowel, Consonant }

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
            ]
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
    }

    #[test]
    fn error() {
        // skipped an index
        assert_matches!{
            Err(Error(ErrorKind::BadPart, _)),
            Part::new(vec![ ((), vec![0, 1, 2, 4]) ]),
        }

        // duplicate, same vec
        assert_matches!{
            Err(Error(ErrorKind::BadPart, _)),
            Part::new(vec![ ((), vec![2, 1, 2, 4]) ]),
        }

        // duplicate, different vec
        assert_matches!{
            Err(Error(ErrorKind::BadPart, _)),
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
            let dp = |x| ::util::DropPusher(drop_history.clone(), x);
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
}
