use ::{Result, ErrorKind};

/// Represents a reordering operation on atoms.
///
/// See the [`Permute`] trait for more information.
///
/// [`Permute`]: trait.Permute.html
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Perm(Vec<u32>);

impl Perm {
    pub fn eye(n: u32) -> Perm
    { Perm((0..n).collect()) }

    pub fn len(&self) -> usize
    { self.0.len() }

    /// Compute the `Perm` that, when applied to the input slice, would sort it.
    pub fn argsort<T: Ord>(xs: &[T]) -> Perm
    {
        let mut perm: Vec<_> = (0..xs.len() as u32).collect();
        perm.sort_by(|&a, &b| xs[a as usize].cmp(&xs[b as usize]));
        Perm(perm)
    }

    /// This performs O(n log n) validation on the data
    /// to verify that it satisfies the invariants of Perm.
    pub fn from_vec(vec: Vec<u32>) -> Result<Perm>
    {Ok({
        ensure!(Self::validate_perm(&vec), ErrorKind::BadPerm);
        Perm(vec)
    })}

    /// This does not check the invariants of Perm.
    ///
    /// # Safety
    ///
    /// `vec` must contain every element in `(0..vec.len())`,
    /// or else the behavior is undefined.
    pub unsafe fn from_vec_unchecked(vec: Vec<u32>) -> Perm
    {
        debug_assert!(Self::validate_perm(&vec));
        Perm(vec)
    }

    // Checks invariants required by Perm for unsafe code.
    fn validate_perm(xs: &[u32]) -> bool
    {
        let mut vec = xs.to_vec();
        vec.sort();
        vec.into_iter().eq(0..xs.len() as u32)
    }

    /// Construct a permutation of length |a| + |b|.
    ///
    /// The inserted elements will be shifted by this permutation's length,
    /// so that they operate on an entirely independent set of data from
    /// the existing elements.
    pub fn append_mut(&mut self, other: &Perm)
    {
        let n = self.0.len() as u32;
        self.0.extend(other.0.iter().map(|&i| i + n));
    }

    #[cfg(test)]
    pub fn random(n: u32) -> Perm
    {
        use ::rand::Rng;

        let mut perm: Vec<_> = (0..n as u32).collect();
        ::rand::thread_rng().shuffle(&mut perm);
        Perm(perm)
    }

    pub fn into_vec(self) -> Vec<u32>
    { self.0 }

    pub fn inverted(&self) -> Perm
    {
        // bah. less code to test...
        Self::argsort(&self.0)
    }

    /// Construct the outer product of self and `slower`, with `self`
    /// being the fast (inner) index.
    ///
    /// The resulting `Perm` will permute blocks of size `self.len()`
    /// according to `slower`, and will permute elements within each
    /// block by `self`.
    pub fn with_outer(&self, slower: &Perm) -> Perm
    {
        assert!((self.len() as u32).checked_mul(slower.len() as u32).is_some());

        let mut perm = Vec::with_capacity(self.len() * slower.len());

        for &block_index in &slower.0 {
            let offset = self.len() as u32 * block_index;
            perm.extend(self.0.iter().map(|&x| x + offset));
        }
        Perm(perm)
    }

    /// Construct the outer product of self and `faster`, with `self`
    /// being the slow (outer) index.
    pub fn with_inner(&self, faster: &Perm) -> Perm
    { faster.with_outer(self) }
}

impl Perm {
    /// Flipped group operator.
    ///
    /// `a.then(b) == b.of(a)`.  The flipped order is more aligned
    /// with this library's generally row-centric design.
    ///
    /// More naturally,
    /// `x.permuted_by(a).permuted_by(b) == x.permuted_by(a.then(b))`.
    pub fn then(&self, other: &Perm) -> Perm
    { self.clone().permuted_by(other) }

    /// Conventional group operator.
    pub fn of(&self, other: &Perm) -> Perm
    { other.then(self) }
}

#[cfg(test)]
pub(crate) fn shuffle<T: Clone>(xs: &[T]) -> (Vec<T>, Perm)
{
    let xs = xs.to_vec();
    let perm = Perm::random(xs.len() as u32);
    (xs.permuted_by(&perm), perm)
}

/// Trait for applying a permutation operation.
///
/// Note that in rsp2, this actually has two roles that are a bit
/// conflated together: (this hasn't caused any trouble *yet*)
///
/// * Permutations of vectors, in general;
///   To this end, there are helper functions like `append` and `with_inner`
///   for constructing and composing permutations, and the implementation
///   of `Permute` uses unsafe code to efficiently and correctly handle
///   types with `Drop` implementations.
///
/// * Permutations of atoms, as a sort of "change of basis operation."
///   Basically, if you have a function `compute_b(Structure) -> B`
///   for some `B` that impls `Permute` (and the type `B` isn't used to
///   represent anything else), then the implementation of `Permute` for
///   `B` likely tries to satisfy the following property:
///
///        compute_b(structure.permute(perm)) == compute_b(structure).permute(perm)
pub trait Permute: Sized {
    // awkward name, but it makes it makes two things clear
    // beyond a shadow of a doubt:
    // - The receiver gets permuted, not the argument.
    //   (relevant when Self is Perm)
    // - The permutation is not in-place.
    fn permuted_by(self, perm: &Perm) -> Self;
}

// (module to protect from lollipop model; the unsafety here
//  is extremely localized)
mod unsafe_impls {
    use super::*;

    impl<T> Permute for Vec<T> {
        fn permuted_by(mut self, perm: &Perm) -> Vec<T>
        {
            use ::std::ptr;

            assert_eq!(self.len(), perm.0.len(),
                "Incorrect permutation length: {} vs {}",
                self.len(), perm.0.len());

            let mut out = Vec::with_capacity(self.len());

            //------------------------------------------------
            // You are now entering a PANIC FREE ZONE

            // Make a bunch of uninitialized elements indexable.
            unsafe { out.set_len(self.len()); }

            for (to, &from) in perm.0.iter().enumerate() {
                // Note: This `as` cast will succeed because all elements
                //       of perm are `< perm.0.len()`, which is a `usize`.
                let tmp = unsafe { ptr::read(&self[from as usize]) };
                unsafe { ptr::write(&mut out[to], tmp) };
            }

            // Don't drop the original items, but do allow the original
            // vec to fall out of scope so the memory can be freed.
            unsafe { self.set_len(0); }

            // Thank you for flying with us. You may now PANIC!
            //------------------------------------------------
            out
        }
    }
}

impl Permute for Perm {
    fn permuted_by(self, perm: &Perm) -> Perm
    { Perm(self.0.permuted_by(perm)) }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::Error;

    #[test]
    fn inverse()
    {
        let perm = Perm::random(20);
        let inv = perm.inverted();

        assert_eq!(perm.clone().permuted_by(&inv), Perm::eye(20));
        assert_eq!(inv.permuted_by(&perm), Perm::eye(20));
    }

    #[test]
    fn invalid() {
        assert_matches!(
            Err(Error(ErrorKind::BadPerm, _)),
            Perm::from_vec(vec![0,1,3,3]));

        assert_matches!(
            Err(Error(ErrorKind::BadPerm, _)),
            Perm::from_vec(vec![1,2,3]));
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn incompatible() {
        // another requirement for the Vec impl's safety
        let _ = vec![4,2,1].permuted_by(&Perm::eye(2));
    }

    #[test]
    fn drop_safety() {
        let (drop_history, dp) = ::util::DropPusher::new_trial();
        {
            let vec = vec![dp(0), dp(1), dp(2), dp(3), dp(4)];

            let vec2 = vec.permuted_by(&Perm(vec![3, 1, 0, 4, 2]));
            assert_eq!(drop_history.borrow().len(), 0);

            drop(vec2);
            assert_eq!(drop_history.borrow().len(), 5);
        }
        assert_eq!(drop_history.borrow().len(), 5);
    }

    #[test]
    fn associativity()
    {
        let xy = Perm::from_vec(vec![1,0,2]).unwrap();
        let zx = Perm::from_vec(vec![2,1,0]).unwrap();
        let xyzx = Perm::from_vec(vec![2,0,1]).unwrap();
        assert_eq!(xy.clone().permuted_by(&zx), xyzx);
        assert_eq!(xy.then(&zx), xyzx);
        assert_eq!(zx.of(&xy), xyzx);
        assert_eq!(
            vec![0,1,2].permuted_by(&xy).permuted_by(&zx),
            vec![0,1,2].permuted_by(&xyzx),
        );
        assert_eq!(
            vec![0,1,2].permuted_by(&xy).permuted_by(&zx),
            vec![2,0,1],
        );
    }

    #[test]
    fn append()
    {
        let mut a = Perm::from_vec(vec![1, 0]).unwrap();
        let b = Perm::from_vec(vec![1, 2, 0]).unwrap();
        a.append_mut(&b);
        assert_eq!(
            vec![00, 01, /* -- */ 10, 11, 12].permuted_by(&a),
            vec![01, 00, /* -- */ 11, 12, 10],
        );
    }

    #[test]
    fn outer()
    {
        let use_outer = |a, b| {
            let a = Perm::from_vec(a).unwrap();
            let b = Perm::from_vec(b).unwrap();
            let xs: Vec<_> =
                (0..b.len()).flat_map(|slow| {
                    (0..a.len()).map(move |fast| 10 * slow + fast)
                }).collect();
            xs.permuted_by(&a.with_outer(&b))
        };

        assert_eq!(
            use_outer(
                vec![1, 0, 2, 3],
                vec![1, 2, 0],
            ),
            vec![
                11, 10, 12, 13,
                21, 20, 22, 23,
                01, 00, 02, 03,
            ],
        );

        // empty perms
        assert_eq!(use_outer(vec![1, 0], vec![]), vec![]);

        assert_eq!(use_outer(vec![], vec![1, 0]), vec![]);
    }
}
