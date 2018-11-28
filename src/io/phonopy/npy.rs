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

// The expansion of do_parse! appears to leave unnecessary parentheses
// around an output expression (the final '>>') that isn't a tuple of
// 2 or more elements.
#![allow(unused_parens)]

use crate::FailResult;
use ::rsp2_kets::Basis;
use ::nom::*;
use ::std::io::Read;
use ::std::mem::size_of;

pub fn read_eigenvector_npy(mut r: impl Read) -> FailResult<Vec<Basis>> {
    let bytes = {
        let mut bytes = vec![];
        r.read_to_end(&mut bytes)?;
        bytes
    };
    self::parse_eigenvector_npy::npy(&bytes)
        .to_full_result()
        .or_else(|e| bail!("{:?}", e)) // generic, not displayable...
}

pub fn read_eigenvalue_npy(mut r: impl Read) -> FailResult<Vec<Vec<f64>>> {
    let bytes = {
        let mut bytes = vec![];
        r.read_to_end(&mut bytes)?;
        bytes
    };
    self::parse_eigenvalue_npy::npy(&bytes)
        .to_full_result()
        .or_else(|e| bail!("{:?}", e)) // generic, not displayable...
}

// macro_rules! noop { ($i:expr) => { tag!($i, []) }; }

// Equivalent of 'pure'/'of' in functional languages.
// Strangely, nom has Option and Result versions but nothing for plain values.
macro_rules! expr { ($i:expr, $e:expr) => { expr_opt!($i, Some($e)) }; }

// This enables you to run another parser on a different input in the middle
//  of a do_parse! chain.
macro_rules! expr_ires { ($i:expr, $e:expr) => { expr_res!($i, $e.to_full_result()) }; }

fn chunk_evenly<T>(slice: &[T], size: usize) -> ::std::slice::Chunks<T> {
    assert_eq!(slice.len() % size, 0);
    slice.chunks(size)
}

use ::nom::digit;
named!(digits<&str>, map_res!(digit, ::std::str::from_utf8));
named!(integer<usize>, map_res!(digits, str::parse::<usize>));

mod parse_eigenvector_npy {
    use super::*;

    // Make no mistake; this file makes no attempt to actually implement the spec,
    //   which contains such phrases as "a [python] object that can be passed
    //   as an argument to the numpy.dtype() constructor".
    // The only aim of this parsing code is to to catch errors when the conditions
    //   diverge from our assumptions.
    named!{blob_size_from_header(&[u8]) -> (usize,usize,usize,usize),
        do_parse!(
            tag!("")
            >> tag!("{'descr': '<c16', 'fortran_order': False, 'shape': (")
            >> nk: integer
            >> tag!(", ")
            >> density: integer
            >> tag!(", ")
            >> na3: integer
            >> tag!(", ")
            >> nb: integer
            >> is_a!(",)} \t\r\n")
            >> eof!()
            >> (nk, density, na3, nb)
        )
    }

    named!{pub npy<Vec<Basis> >,
        do_parse!(
            tag!("")
            >> tag!([0x93])
            >> tag!("NUMPY")
            >> major: le_u8
            >> minor: le_u8
            >> header_size: call!(|s| match (major, minor) {
                (1, 0) => map!(s, le_u16, |x| x as usize),
                (2, 0) => map!(s, le_u32, |x| x as usize),
                _ => panic!("unsupported NPY version {}.{}", major, minor)
            })
            >> header: take!(header_size)

            >> dims_tup: expr_ires!(call!(header, blob_size_from_header))
            >> nk:  expr!(dims_tup.0 * dims_tup.1)
            >> na3: expr!(dims_tup.2)
            >> nb:  expr!(dims_tup.3)

            // the order of the sizes written here reflects the indices:
            //  kpoint, (atom * axis), band, (real | imag)
            >> blob_size: expr!(nk * na3 * nb * 2 * size_of::<f64>())
            >> blob: take!(blob_size)
            >> eof!()

            >> ({
                let mut floats = vec![0f64; blob_size/size_of::<f64>()];
                read_f64v_le(&mut floats, blob);
                let floats = floats;

                // move atoms to the last dimension

                let make_strides = |(d,c,b,a)| (c*b*a, b*a, a, 1);
                let dot4 = |(a1,a2,a3,a4), (b1,b2,b3,b4)| a1*b1 + a2*b2 + a3*b3 + a4*b4;

                let mut out = vec![0f64; floats.len()];
                let old_to_new = |(k,a,b,i)| (k,b,i,a);
                let new_to_old = |(k,b,i,a)| (k,a,b,i);
                let dims_old = (nk, na3, nb, 2);
                let strides_old = make_strides(dims_old);
                let strides_new = new_to_old(make_strides(old_to_new(dims_old)));
                let index_old = |pos| dot4(strides_old, pos);
                let index_new = |pos| dot4(strides_new, pos);
                for k in 0..nk {
                    for a in 0..na3 {
                        for b in 0..nb {
                            for i in 0..2 {
                                let pos = (k,a,b,i);
                                out[index_new(pos)] = floats[index_old(pos)];
                            }
                        }
                    }
                }

                // collect for each kpoint
                let bases: Vec<_> =
                    chunk_evenly(&out, strides_new.0)
                    .map(|c| Basis::new(c.to_owned(), nb))
                    .collect();
                assert_eq!(bases.len(), nk);
                bases
            })
        )
    }
}

// FIXME shameful copypasta
// the only tangible difference is that:
//   * eigenvector.npy is shape (l, m, n, n, 2)
//   * eigenvalue.npy  is shape (l, m, n)
mod parse_eigenvalue_npy {
    use super::*;

    // Make no mistake; this file makes no attempt to actually implement the spec,
    //   which contains such phrases as "a [python] object that can be passed
    //   as an argument to the numpy.dtype() constructor".
    // The only aim of this parsing code is to to catch errors when the conditions
    //   diverge from our assumptions.
    named!{blob_size_from_header(&[u8]) -> (usize,usize,usize),
        do_parse!(
            tag!("")
            >> tag!("{'descr': '<f8', 'fortran_order': False, 'shape': (")
            >> nk: integer
            >> tag!(", ")
            >> density: integer
            // >> tag!(", ")
            // >> na3: integer
            >> tag!(", ")
            >> nb: integer
            >> is_a!(",)} \t\r\n")
            >> eof!()
            >> (nk, density, nb)
        )
    }

    named!{pub npy<Vec<Vec<f64> > >,
        do_parse!(
            tag!("")
            >> tag!([0x93])
            >> tag!("NUMPY")
            >> major: le_u8
            >> minor: le_u8
            >> header_size: call!(|s| match (major, minor) {
                (1, 0) => map!(s, le_u16, |x| x as usize),
                (2, 0) => map!(s, le_u32, |x| x as usize),
                _ => panic!("unsupported NPY version {}.{}", major, minor)
            })
            >> header: take!(header_size)

            >> dims_tup: expr_ires!(call!(header, blob_size_from_header))
            >> nk:  expr!(dims_tup.0 * dims_tup.1)
            >> nb:  expr!(dims_tup.2)

            // the order of the sizes written here reflects the indices:
            //  kpoint, band
            >> blob_size: expr!(nk * nb * size_of::<f64>())
            >> blob: take!(blob_size)
            >> eof!()

            >> ({
                let mut floats = vec![0f64; blob_size/size_of::<f64>()];
                read_f64v_le(&mut floats, blob);
                let floats = floats;

                // no need to reorder axes
                let out = floats;

                // let make_strides = |(d,c,b,a)| (c*b*a, b*a, a, 1);
                // let dot4 = |(a1,a2,a3,a4), (b1,b2,b3,b4)| a1*b1 + a2*b2 + a3*b3 + a4*b4;

                // let mut out = vec![0f64; floats.len()];
                // let old_to_new = |(k,a,b,i)| (k,b,i,a);
                // let new_to_old = |(k,b,i,a)| (k,a,b,i);
                // let dims_old = (nk, na3, nb, 2);
                // let strides_old = make_strides(dims_old);
                // let strides_new = new_to_old(make_strides(old_to_new(dims_old)));
                // let index_old = |pos| dot4(strides_old, pos);
                // let index_new = |pos| dot4(strides_new, pos);
                // for k in 0..nk {
                //     for a in 0..na3 {
                //         for b in 0..nb {
                //             for i in 0..2 {
                //                 let pos = (k,a,b,i);
                //                 out[index_new(pos)] = floats[index_old(pos)];
                //             }
                //         }
                //     }
                // }

                let at_each_qpoint: Vec<_> =
                    chunk_evenly(&out, nb)
                    .map(|slc| slc.to_vec())
                    .collect();
                assert_eq!(at_each_qpoint.len(), nk);
                at_each_qpoint
            })
        )
    }
}

/// Read a vector of bytes into a vector of f64s. The values are read in
/// little-endian format.
fn read_f64v_le(dst: &mut [f64], src: &[u8]) {
    let mut u64s = vec![0u64; dst.len()];

    ::byte_tools::read_u64v_le(&mut u64s, src);

    for (f, i) in dst.iter_mut().zip(u64s) {
        *f = f64::from_bits(i);
    }
}
