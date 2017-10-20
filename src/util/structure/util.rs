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

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Perm(Vec<u32>);

    impl Perm {
        pub fn eye(n: u32) -> Perm
        { Perm((0..n).collect()) }

        pub fn len(&self) -> usize { self.0.len() }

        /// This performs O(n log n) validation on the data
        /// to verify that it satisfies the invariants of Perm.
        pub fn from_vec(vec: Vec<u32>) -> Result<Perm>
        {Ok({
            ensure!({
                let mut x = vec.clone();
                x.sort();
                x.into_iter().eq(0..vec.len() as u32)
            }, ErrorKind::BadPerm);

            Perm(vec)
        })}

        /// This does not check the invariants of Perm,
        /// namely that vec must contain each element
        /// in `(0..vec.len())` once.
        ///
        /// This is unsafe because Perm may in the future
        /// use unsafe optimizations based on these invariants.
        // (NOTE: such as moving Drop data without clones)
        pub unsafe fn from_vec_unchecked(vec: Vec<u32>) -> Perm
        {
            debug_assert!({
                let mut x = vec.clone();
                x.sort();
                x.into_iter().eq(0..vec.len() as u32)
            });

            Perm(vec)
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
        fn permuted_by(self, perm: &Perm) -> Self;
    }

    impl<T> Permute for Vec<T> {
        fn permuted_by(self, perm: &Perm) -> Vec<T>
        {
            use ::std::ptr;

            assert_eq!(self.len(), perm.0.len());
            let mut out = Vec::with_capacity(self.len());

            //------------------------------------------------
            // You are now entering a PANIC FREE ZONE
            unsafe { out.set_len(self.len()); }
            for (to, &from) in perm.0.iter().enumerate() {
                let tmp = unsafe { ptr::read(&self[from as usize]) };
                unsafe { ptr::write(&mut out[to], tmp) };
            }
            ::std::mem::forget(self);
            //------------------------------------------------

            out
        }
    }

    impl Permute for Perm {
        fn permuted_by(self, perm: &Perm) -> Perm
        { Perm(self.0.permuted_by(perm)) }
    }

    #[cfg(test)]
    #[deny(unused)]
    mod tests {
        use super::{Perm, Permute};

        #[test]
        fn perm_inverse()
        {
            let perm = Perm::random(20);
            let inv = perm.inverted();

            assert_eq!(perm.clone().permuted_by(&inv), Perm::eye(20));
            assert_eq!(inv.permuted_by(&perm), Perm::eye(20));
        }
    }
}
