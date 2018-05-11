#![feature(step_by)]
use std::cmp;
use std::ops::Mul;
use std::fmt::Debug;

use vec::SparseVec;
use slice::SparseSlice;
use iter::SparseIterator;

#[cfg(test)]
use test::Bencher;

fn old_pointwise_mul<T>(a: SparseSlice<T>, b: SparseSlice<T>) -> SparseVec<T>
 where T: Mul<T,Output=T> + Debug + Clone,
{
	assert_eq!(a.dim(), b.dim());
	let dim = a.dim();

	// lolidunno haaalp
	let mut ait = a.sparse_iter().sparse_cloned();
	let mut bit = b.sparse_iter().sparse_cloned();

	if let (Some((i0,a0)),Some((k0,b0))) = (ait.next(),bit.next()) {
		let cap = cmp::min(a.nnz(), b.nnz());
		let mut outval = Vec::with_capacity(cap);
		let mut outpos = Vec::with_capacity(cap);
//		let mut outval = Vec::new();
//		let mut outpos = Vec::new();

		let (mut icur,mut acur) = (i0,a0);
		let (mut kcur,mut bcur) = (k0,b0);
		'a: loop {
			// scan to an index present in both vectors
			while icur < kcur {
				if let Some((i,a)) = ait.next() { acur = a; icur = i; }
				else { break 'a; } // stop immediately if either iterator ends
			}
			debug_assert!(icur < dim, "pos exceeds dimension");
			while kcur < icur {
				if let Some((k,b)) = bit.next() { bcur = b; kcur = k; }
				else { break 'a; }
			}
			debug_assert!(kcur < dim, "pos exceeds dimension");

			// at this point, kcur >= icur
			if icur == kcur {
				// these clones are not strictly necessary, but there's no way for the compiler
				// to see that given the way this is written
				// (which is based on the precondition that the indices are sorted)
				outval.push(acur.clone() * bcur.clone());
				outpos.push(icur);
				if let Some((i,a)) = ait.next() { acur = a; icur = i; }
				else { break 'a; }
			}
			assert!(icur != kcur);
		}

		SparseVec::from_parts_unchecked(dim, outval, outpos)

	// at least one iterator had no elements; one of the vectors is the zero vector
	} else {
		SparseVec::zero(dim)
	}
}

fn new_pointwise_mul<T>(a: SparseSlice<T>, b: SparseSlice<T>) -> SparseVec<T>
 where T: Mul<T,Output=T> + Debug + Clone,
{
	assert_eq!(a.dim(), b.dim());
	let ait = a.sparse_iter().sparse_cloned();
	let bit = b.sparse_iter().sparse_cloned();
	let it = ait.sparse_intersection(bit)
	   .sparse_map(|(x,y)| x * y);
	SparseVec::from_unshaped_sparse_iter(a.dim(),it)
}


// NOTE: Seems that the cost of allocating the vectors far outweighs the cost of performing the
//   multiplication.  When `old_pointwise_mul` uses `Vec::new()` for `newval` and `newpos`:
//
//test bench::math::bench_old_matching          ... bench:         277 ns/iter (+/- 12)
//test bench::math::bench_new_matching          ... bench:         293 ns/iter (+/- 25)
//test bench::math::bench_old_nonmatching       ... bench:          84 ns/iter (+/- 1)
//test bench::math::bench_new_nonmatching       ... bench:          99 ns/iter (+/- 2)
//test bench::math::bench_old_one_match         ... bench:         124 ns/iter (+/- 1)
//test bench::math::bench_new_one_match         ... bench:         142 ns/iter (+/- 2)
//
// When `old_pointwise_mul` uses `Vec::with_capacity(cap)`:
//
//test bench::math::bench_old_matching          ... bench:         136 ns/iter (+/- 2)
//test bench::math::bench_new_matching          ... bench:         290 ns/iter (+/- 3)
//test bench::math::bench_old_nonmatching       ... bench:         111 ns/iter (+/- 1)
//test bench::math::bench_new_nonmatching       ... bench:          98 ns/iter (+/- 1)
//test bench::math::bench_old_one_match         ... bench:         113 ns/iter (+/- 3)
//test bench::math::bench_new_one_match         ... bench:         140 ns/iter (+/- 5)


#[bench]
fn bench_old_matching(b: &mut Bencher) {
	let v = SparseVec::from_parts(32, vec![2.;13], (0..25).step_by(2).collect());
	let u = SparseVec::from_parts(32, vec![2.;13], (0..25).step_by(2).collect());
	b.iter(|| old_pointwise_mul(v.slice(0..32), u.slice(0..32)) )
}

#[bench]
fn bench_new_matching(b: &mut Bencher) {
	let v = SparseVec::from_parts(32, vec![2.;13], (0..25).step_by(2).collect());
	let u = SparseVec::from_parts(32, vec![2.;13], (0..25).step_by(2).collect());
	b.iter(|| new_pointwise_mul(v.slice(0..32), u.slice(0..32)) )
}

#[bench]
fn bench_old_nonmatching(b: &mut Bencher) {
	let v = SparseVec::from_parts(32, vec![2.;13], (0..25).step_by(2).collect());
	let u = SparseVec::from_parts(32, vec![2.;13], (1..26).step_by(2).collect());
	b.iter(|| old_pointwise_mul(v.slice(0..32), u.slice(0..32)) )
}

#[bench]
fn bench_new_nonmatching(b: &mut Bencher) {
	let v = SparseVec::from_parts(32, vec![2.;13], (0..25).step_by(2).collect());
	let u = SparseVec::from_parts(32, vec![2.;13], (1..26).step_by(2).collect());
	b.iter(|| new_pointwise_mul(v.slice(0..32), u.slice(0..32)) )
}
#[bench]
fn bench_old_one_match(b: &mut Bencher) {
	let v = SparseVec::from_parts(32, vec![2.;13], (0..23).step_by(2).chain(::std::iter::once(30)).collect());
	let u = SparseVec::from_parts(32, vec![2.;13], (1..24).step_by(2).chain(::std::iter::once(30)).collect());
	b.iter(|| old_pointwise_mul(v.slice(0..32), u.slice(0..32)) )
}

#[bench]
fn bench_new_one_match(b: &mut Bencher) {
	let v = SparseVec::from_parts(32, vec![2.;13], (0..23).step_by(2).chain(::std::iter::once(30)).collect());
	let u = SparseVec::from_parts(32, vec![2.;13], (1..24).step_by(2).chain(::std::iter::once(30)).collect());
	b.iter(|| new_pointwise_mul(v.slice(0..32), u.slice(0..32)) )
}
