// FIXME kill these once there's utilities that support these
//       operations on variable length slices/vecs
#[cfg(test)]
use ::ordered_float::NotNaN;

pub(crate) fn dot_n3_33(coords: &[[f64; 3]], mat: &[[f64; 3]; 3]) -> Vec<[f64; 3]>
{
    use ::rsp2_array_utils::dot;
    coords.iter().map(|v| dot(v, mat)).collect()
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
