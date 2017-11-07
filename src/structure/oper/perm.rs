use ::{Result, ErrorKind};

/// Represents a reordering operation on atoms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Perm(Vec<u32>);

impl Perm {
    #[allow(unused)]
    pub fn eye(n: u32) -> Perm
    { Perm((0..n).collect()) }

    #[allow(unused)]
    pub fn len(&self) -> usize
    { self.0.len() }

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
    #[allow(unused)]
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
    #[allow(unused)] // FIXME test
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

    #[allow(unused)]
    pub fn into_vec(self) -> Vec<u32>
    { self.0 }

    pub fn inverted(&self) -> Perm
    {
        // bah. less code to test...
        argsort(&self.0)
    }
}

#[allow(unused)]
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

/// Decompose a sequence into (sorted, perm).
///
/// The output will satisfy `sorted.permute_by(&perm) == original`
pub(crate) fn sort<T: Clone + Ord>(xs: &[T]) -> (Vec<T>, Perm)
{
    let mut xs: Vec<_> = xs.iter().cloned().enumerate().collect();
    xs.sort_by(|a,b| a.1.cmp(&b.1));
    let (perm, xs) = xs.into_iter().map(|(i, x)| (i as u32, x)).unzip();
    (xs, Perm(perm))
}

#[cfg(test)]
pub(crate) fn shuffle<T: Clone>(xs: &[T]) -> (Vec<T>, Perm)
{
    let xs = xs.to_vec();
    let perm = Perm::random(xs.len() as u32);
    (xs.permuted_by(&perm), perm)
}

pub(crate) fn argsort<T: Clone + Ord>(xs: &[T]) -> Perm
{ sort(xs).1 }

pub(crate)
trait Permute: Sized {
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
    fn perm_inverse()
    {
        let perm = Perm::random(20);
        let inv = perm.inverted();

        assert_eq!(perm.clone().permuted_by(&inv), Perm::eye(20));
        assert_eq!(inv.permuted_by(&perm), Perm::eye(20));
    }

    #[test]
    fn test_invalid_perm() {
        assert_matches!(
            Err(Error(ErrorKind::BadPerm, _)),
            Perm::from_vec(vec![0,1,3,3]));

        assert_matches!(
            Err(Error(ErrorKind::BadPerm, _)),
            Perm::from_vec(vec![1,2,3]));
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn test_incompatible_perm() {
        // another requirement for the Vec impl's safety
        let _ = vec![4,2,1].permuted_by(&Perm::eye(2));
    }

    #[test]
    fn test_permute_drop() {
        use ::std::rc::Rc;
        use ::std::cell::RefCell;
        let drop_history = Rc::new(RefCell::new(vec![]));

        struct DropPusher(Rc<RefCell<Vec<u32>>>, u32);
        impl Drop for DropPusher {
            fn drop(&mut self) {
                self.0.borrow_mut().push(self.1);
            }
        }

        {
            let dp = |x| DropPusher(drop_history.clone(), x);
            let vec = vec![dp(0), dp(1), dp(2), dp(3), dp(4)];

            let vec2 = vec.permuted_by(&Perm(vec![3, 1, 0, 4, 2]));
            assert_eq!(drop_history.borrow().len(), 0);

            drop(vec2);
            assert_eq!(drop_history.borrow().len(), 5);
        }
        assert_eq!(drop_history.borrow().len(), 5);
    }

    #[test]
    fn permute_associativity()
    {
        let xy = Perm::from_vec(vec![1,0,2]).unwrap();
        let zx = Perm::from_vec(vec![2,1,0]).unwrap();
        let xyzx = Perm::from_vec(vec![2,0,1]).unwrap();
        assert_eq!(xy.clone().permuted_by(&zx), xyzx);
        assert_eq!(xy.then(&zx), xyzx);
        assert_eq!(zx.of(&xy), xyzx);
        assert_eq!(
            vec![0,1,2].permuted_by(&xy).permuted_by(&zx),
            vec![0,1,2].permuted_by(&xyzx));
        assert_eq!(
            vec![0,1,2].permuted_by(&xy).permuted_by(&zx),
            vec![2,0,1]);
    }
}
