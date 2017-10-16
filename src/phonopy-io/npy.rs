use ::Result;
use ::std::io::Read;

use ::rsp2_kets::Basis;

pub fn read_eigenvector_npy<R: Read>(mut r: R) -> Result<Vec<Basis>> {
    let bytes = {
        let mut bytes = vec![];
        r.read_to_end(&mut bytes)?;
        bytes
    };
    self::parse_eigenvector_npy::npy(&bytes)
        .to_full_result()
        .or_else(|e| bail!("{:?}", e)) // generic, not displayable...
}

pub fn read_eigenvalue_npy<R: Read>(mut r: R) -> Result<Vec<Vec<f64>>> {
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
    use ::nom::*;
    use ::std::mem::size_of;
    use ::rsp2_kets::Basis;

    use super::{chunk_evenly, integer};

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
                ::byte_tools::read_f64v_le(&mut floats, blob);
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
    use ::nom::*;
    use ::std::mem::size_of;

    use super::{chunk_evenly, integer};

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
                ::byte_tools::read_f64v_le(&mut floats, blob);
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

//-----------------
// git format-patch for the phonopy eigenvector hack

/*
From 62ed90a875fdbda1181a3a7ebc8192a850765266 Mon Sep 17 00:00:00 2001
From: Michael Lamparski <diagonaldevice@gmail.com>
Date: Thu, 27 Apr 2017 08:59:38 -0400
Subject: [PATCH] "the phonopy .npy eigenvector hack" (re-redux)

a hack that causes phonopy to forget about band.yaml and
instead dump some npy files

You may summon the abomination on-demand with this chant:

    EIGENVECTOR_NPY_HACK=1 phonopy [phonopy args]

This is a rebase and squash after diverging from upstream for
several months, with an additional fix to output the correct
object for eigenvalues.npy.
---
 phonopy/phonon/band_structure.py | 16 ++++++++++++++++
 1 file changed, 16 insertions(+)

diff --git a/phonopy/phonon/band_structure.py b/phonopy/phonon/band_structure.py
index 900ec3b..f004dd5 100644
--- a/phonopy/phonon/band_structure.py
+++ b/phonopy/phonon/band_structure.py
@@ -161,6 +161,22 @@ class BandStructure(object):
             text.append('')
             w.write("\n".join(text))

+            # "The eigenvector hack"
+            # Drop everything we're doing, write a binary file, and, most importantly,
+            # do NOT waste time serializing the vectors to yaml.
+            import os
+            if os.getenv('EIGENVECTOR_NPY_HACK'):
+                if self._eigenvectors:
+                    np.save('eigenvector.npy', self._eigenvectors)
+                if self._distances:
+                    np.save('q-distance.npy', self._distances)
+                np.save('eigenvalue.npy', self._frequencies)
+
+                # don't leave behind an incomplete yaml file
+                w.close()
+                os.unlink(w.name)
+                return
+
             for i in range(len(self._paths)):
                 qpoints = self._paths[i]
                 distances = self._distances[i]
--
2.14.1
*/
