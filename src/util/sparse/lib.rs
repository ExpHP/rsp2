//#![feature(test)]

// FIXME: I copied this almost directly from an older project,
//        written back before I think even rust was 1.0.
//
//        THE VAST MAJORITY OF THE CODE IS NOT USED BY RSP2.
//        Chances are that A LOT of stuff can be ripped out.
//
//        It has been a long time since I looked at a lot of this
//        code, and it had a large API surface. It's possible that
//        I may regret this decision.

#[cfg(test)]
#[macro_use]
extern crate rsp2_assert_close;

extern crate rustc_test as test;
extern crate num_traits;

#[macro_use]
mod macros;

mod iter;
pub mod mat;
mod slice;
mod traits;
mod vec;
// math has some output result types that need to be somehow visible.
pub mod math;

pub use iter::IntoSparseIterator;
pub use iter::ShapedSparseIterator;
pub use iter::SortedSparseIterator;
pub use iter::SparseIterator;
pub use iter::UnionValue;
pub use iter::UniqueSparseIterator;
pub use math::MatMatMul;
pub use math::SparseDenseMath;
pub use math::SparseSparseMath;
pub use slice::SparseSlice;
pub use traits::DenseIndex;
pub use traits::Shape;
pub use vec::SparseVec;
