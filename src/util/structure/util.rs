// FIXME kill these once there's utilities that support these
//       operations on variable length slices/vecs
#[cfg(test)]
use ::ordered_float::NotNaN;

// Multiply on the right
pub(crate) fn dot_n3_33(coords: &[[f64; 3]], mat: &[[f64; 3]; 3]) -> Vec<[f64; 3]>
{
    use ::rsp2_array_utils::dot;
    coords.iter().map(|v| dot(v, mat)).collect()
}

// Multiply by transpose on the right
//
// I think this one is more likely to be able to use SIMD
// but I have not tested this. - ML
#[allow(non_snake_case)]
pub(crate) fn dot_n3_33T(coords: &[[f64; 3]], mat: &[[f64; 3]; 3]) -> Vec<[f64; 3]>
{
    use ::rsp2_array_utils::vec_from_fn;
    coords.iter().map(|v|
        vec_from_fn(|c| (0..3).map(|k| v[k] * mat[c][k]).sum())
    ).collect()
}

pub(crate) fn translate_mut_n3_3(coords: &mut [[f64; 3]], t: &[f64; 3])
{
    for row in coords {
        for k in 0..3 {
            row[k] += t[k];
        }
    }
}

pub(crate) fn translate_mut_n3_n3(coords: &mut [[f64; 3]], by: &[[f64; 3]])
{
    assert_eq!(coords.len(), by.len());
    for i in 0..coords.len() {
        for k in 0..3 {
            coords[i][k] += by[i][k];
        }
    }
}

#[cfg(test)]
pub(crate) fn not_nan_n3(coords: Vec<[f64; 3]>) -> Vec<[NotNaN<f64>; 3]> {
    use ::slice_of_array::prelude::*;
    // still a newtype?
    assert_eq!(::std::mem::size_of::<f64>(), ::std::mem::size_of::<NotNaN<f64>>());
    // (NotNaN has undefined behavior for NaN so we must check)
    assert!(coords.flat().iter().all(|x| !x.is_nan()));
    unsafe { ::std::mem::transmute(coords) }
}

#[cfg(test)]
pub(crate) fn eq_unordered_n3(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
    let mut a = not_nan_n3(a.to_vec()); a.sort();
    let mut b = not_nan_n3(b.to_vec()); b.sort();
    a == b
}

#[allow(dead_code)] // FIXME
pub(crate) mod perm {
    use ::{Result, ErrorKind};

    /// Represents a reordering operation on atoms.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Perm(Vec<u32>);

    impl Perm {
        pub fn eye(n: u32) -> Perm
        { Perm((0..n).collect()) }

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
            argsort(&self.0)
        }
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
    }
}
