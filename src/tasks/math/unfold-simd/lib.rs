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

//struct SimdAligned<Xs, V> {
//    excess: V,
//    excess_len: usize,
//    // invariant: aligned to mem::align_of::<V>()
//    // invariant: contains a whole number of Vs
//    data: [X],
//}

extern crate rsp2_kets;
extern crate rsp2_soa_ops;
extern crate faster;
#[macro_use]
extern crate itertools;

use rsp2_kets::{Ket, AsKetRef, KetRef};
use faster::prelude::*;
use std::ops;
use itertools::{iterate};
use rsp2_soa_ops::{Part, Partition};

pub struct SimdGammaUnfolder {
    /// Normalized kets corresponding to plane waves at `[1,0,0]`, `[0,1,0]`, and `[0,0,1]` in
    /// the supercell reciprocal space.
    pub sc_q_kets: [Ket; 3],
    /// Normalized kets corresponding to plane waves at `[1,0,0]`, `[0,1,0]`, and `[0,0,1]` in
    /// the primitive cell reciprocal space.
    pub pc_q_kets: [Ket; 3],

    /// Normalized ket corresponding to the q point sampled with the most negative index of
    /// all sampled primitive reciprocal points along each axis.
    pub initial_pc_q_ket: Ket,

    /// Index of each q point in output.
    ///
    /// You could use this to control the order of the output. (TODO: not implemented)
    pub sc_indices: Vec<[u32; 3]>,

    /// Number of images sampled along each primitive cell reciprocal axis.
    pub pc_dims: [u32; 3],
    /// Number of q points along each supercell reciprocal axis that are distinct under the
    /// primitive cell reciprocal lattice.
    pub sc_dims: [u32; 3],
}

impl SimdGammaUnfolder {
    pub fn unfold<K: AsKetRef>(&self, eigenvector: K) -> Vec<f64>
    { unfold(self, eigenvector.as_ket_ref()) }
}

#[inline(never)]
fn unfold(
    unfolder: &SimdGammaUnfolder,
    eigenvector: KetRef<'_>,
) -> Vec<f64>
{
    let SimdGammaUnfolder {
        sc_q_kets, pc_q_kets, initial_pc_q_ket, sc_indices, pc_dims, sc_dims,
    } = unfolder;
    // faster implements `zip` for up to 12 iterators.
    //
    // Luckily, for twisted BLG (where the second two periods are always trivial), 12 iterators
    // is all we need.
    assert_eq!(sc_dims[1], 1, "unfold-simd is currently limited to SC dims of the form [n, 1, 1]");
    assert_eq!(sc_dims[2], 1, "unfold-simd is currently limited to SC dims of the form [n, 1, 1]");

    // TODO simple fix with Perm::argsort and composition
    assert_eq!(&sc_indices[..], &gen_q_indices(*sc_dims)[..], "reordering output not yet implemented");

    let axis_part; // declared out here for a longer lifetime
    let axis_evs = {
        let labels = [0, 1, 2].iter().cloned().cycle().take(eigenvector.len());
        axis_part = Part::from_ord_keys(labels);

        let reals = eigenvector.real().to_vec().into_unlabeled_partitions(&axis_part);
        let imags = eigenvector.imag().to_vec().into_unlabeled_partitions(&axis_part);
        izip!(reals, imags).map(|(r, i)| Ket::new(r, i))
    };
    assert_eq!(axis_evs.len(), 3);

    let dim_sc_total = sc_dims.iter().product::<u32>() as usize;

    let mut probs = vec![0.0; dim_sc_total];

    // different axes work together (the terms they contribute cannot cancel each other)
    for axis_ev in axis_evs {
        let mut iter = (
            sc_q_kets[0].real().simd_iter(f64s(0.0)),
            sc_q_kets[0].imag().simd_iter(f64s(0.0)),
            pc_q_kets[0].real().simd_iter(f64s(0.0)),
            pc_q_kets[0].imag().simd_iter(f64s(0.0)),
            pc_q_kets[1].real().simd_iter(f64s(0.0)),
            pc_q_kets[1].imag().simd_iter(f64s(0.0)),
            pc_q_kets[2].real().simd_iter(f64s(0.0)),
            pc_q_kets[2].imag().simd_iter(f64s(0.0)),
            initial_pc_q_ket.real().simd_iter(f64s(0.0)),
            initial_pc_q_ket.imag().simd_iter(f64s(0.0)),
            // note: the zeros here guarantee that the SIMD remainder does not add anything
            //       artificial to the sums.
            axis_ev.real().simd_iter(f64s(0.0)),
            axis_ev.imag().simd_iter(f64s(0.0)),
        ).zip();

        // cancellation can occur within an axis (as dot products are performed)
        let mut dot_prods = vec![Complex::zero(); dim_sc_total];

        let [dim_a, dim_b, dim_c] = *pc_dims;
        let [dim_s, _, _] = *sc_dims;
        iter.simd_do_each(|(sr, si, ar, ai, br, bi, cr, ci, ir, ii, vr, vi)| {
            let s = Complex(sr, si);
            let a = Complex(ar, ai);
            let b = Complex(br, bi);
            let c = Complex(cr, ci);
            let init = Complex(ir, ii);
            let ev = Complex(vr, vi);

            // different PC reciprocal lattice vectors work together.
            //
            // phases are produced by iterated multiplication. This should introduce approx
            // `O(sqrt(N) * epsilon)` error near the end of each axis of length N, but it is quite
            // fast and saves memory.
            let phase = init;
            for phase in iterate(phase, |&p| p * a).take(dim_a as usize) {
                for phase in iterate(phase, |&p| p * b).take(dim_b as usize) {
                    for phase in iterate(phase, |&p| p * c).take(dim_c as usize) {

                        // different SC reciprocal lattice vectors compete with each other.
                        let phases = iterate(phase, |&p| p * s).take(dim_s as usize);

                        //---------------------------------------------------
                        // TODO: Investigate doing this in chunks.
                        //
                        //       By iterating over the entire `dot_prods` vector here, I suspect
                        //       that we risk evicting all of the vectors in the SIMD zip iterator
                        //       out of L1 cache (at least, on CPUs that use a LRU heuristic).
                        //---------------------------------------------------
                        // TODO: Investigate threading here.
                        //
                        //       This currently seems to be the best place for threads.  If we do
                        //       chunking here, however, I doubt that threading will help any
                        //       further, and then I'm not sure where we could put threads.
                        //---------------------------------------------------
                        assert_eq!(dim_s as usize, dot_prods.len());
                        for (dot, phase) in izip!(&mut dot_prods, phases) {
                            // note: the zero default value of ev guarantees that nothing artificial
                            //       is added to the sum by the SIMD remainder.
                            *dot = *dot + phase.conj() * ev;
                        } // s
                    } // c
                } // b
            } // a
        }); // SIMD iter

        assert_eq!(probs.len(), dot_prods.len());
        for (prob, dot) in izip!(&mut probs, dot_prods) {
            *prob = *prob + dot.sum().sqnorm();
        }
    }

    let total: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= total);
    probs
}

fn gen_q_indices(sc_dims: [u32; 3]) -> Vec<[u32; 3]> {
    assert_eq!(sc_dims[1], 1, "unfold-simd is currently limited to SC dims of the form [n, 1, 1]");
    assert_eq!(sc_dims[2], 1, "unfold-simd is currently limited to SC dims of the form [n, 1, 1]");
    (0..sc_dims[0]).map(|i| [i, 0, 0]).collect()
}

trait Real
    : Copy
    + ops::Add<Output=Self>
    + ops::Mul<Output=Self>
    + ops::Sub<Output=Self>
    + ops::Neg<Output=Self>
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl Real for f64 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 0.0 }
}

impl Real for f64s {
    #[inline] fn zero() -> Self { f64s(0.0) }
    #[inline] fn one() -> Self { f64s(0.0) }
}

#[derive(Debug, Copy, Clone)]
struct Complex<V>(V, V);
impl<V: Real> Complex<V> {
    #[inline(always)] fn zero() -> Self { Complex(V::zero(), V::zero()) }
    #[inline(always)] fn conj(self) -> Self { Complex(self.0, -self.1) }
    #[inline(always)] fn sqnorm(self) -> V { self.0 * self.0 + self.1 * self.1 }
}

impl Complex<f64s> {
    #[inline(always)] fn sum(self) -> Complex<f64> { Complex(self.0.sum(), self.1.sum()) }
}

impl<V: Real> ops::Add for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Complex<V>) -> Self
    { Complex(self.0 + other.0, self.1 + other.1) }
}

impl<V: Real> ops::Mul for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Complex<V>) -> Self
    { Complex(self.0 * other.0 - self.1 * other.1, self.0 * other.1 + self.1 * other.0) }
}
