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

// FIXME kill these once there's utilities that support these
//       operations on variable length slices/vecs
#[cfg(test)]
use ordered_float::NotNan;

use crate::IntPrecisionError;

use rsp2_array_types::{V3, M33, M3};

pub(crate) fn translate_mut_n3_3(coords: &mut [V3], t: &V3)
{
    for row in coords {
        *row += t;
    }
}

pub(crate) fn translate_mut_n3_n3(coords: &mut [V3], by: &[V3])
{
    assert_eq!(coords.len(), by.len());
    izip!(coords, by).for_each(|(a, b)| *a += b);
}

#[cfg(test)]
pub(crate) fn not_nan_n3(coords: Vec<V3>) -> Vec<V3<NotNan<f64>>> {
    // still a newtype?
    assert_eq!(::std::mem::size_of::<f64>(), std::mem::size_of::<NotNan<f64>>());
    // (NotNan has undefined behavior for NaN so we must check)
    assert!(coords.iter().flat_map(|v| v).all(|x| !x.is_nan()));
    unsafe { std::mem::transmute(coords) }
}

#[cfg(test)]
pub(crate) fn eq_unordered_n3(a: &[V3], b: &[V3]) -> bool {
    let mut a = not_nan_n3(a.to_vec()); a.sort();
    let mut b = not_nan_n3(b.to_vec()); b.sort();
    a == b
}

// these f64 -> i32 conversions are written on a silly little type
// simply to avoid having a function with a signature like 'fn f(x: f64, tol: f64)'
// where the arguments could be swapped
pub(crate) struct Tol(pub(crate) f64);
#[allow(unused)]
impl Tol {
    pub(crate) fn unfloat(&self, x: f64) -> Result<i32, IntPrecisionError>
    {Ok({
        let r = x.round();
        if (r - x).abs() > self.0 {
            return Err(IntPrecisionError {
                backtrace: failure::Backtrace::new(),
                value: x,
            });
        }
        r as i32
    })}

    pub(crate) fn unfloat_v3(&self, v: &V3) -> Result<V3<i32>, IntPrecisionError>
    { v.try_map(|x| self.unfloat(x)) }

    pub(crate) fn unfloat_m33(&self, m: &M33) -> Result<M33<i32>, IntPrecisionError>
    { rsp2_array_utils::try_map_arr(m.0, |v| self.unfloat_v3(&v)).map(M3) }
}

#[inline(always)] // hopefully encourage LLVM to do crazy bit math for constant moduli
#[cfg(feature = "nightly")]
#[inline(always)]
pub(crate) fn mod_euc(a: i32, b: i32) -> i32 { i32::mod_euc(a, b) }

#[inline(always)] // hopefully encourage LLVM to do crazy bit math for constant moduli
#[cfg(not(feature = "nightly"))]
pub(crate) fn mod_euc(a: i32, b: i32) -> i32 {
    match a % b {
        r if r < 0 => r + i32::abs(b),
        r => r,
    }
}
