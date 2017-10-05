//! read-tools from crates.io extended with float functions.
//!
//! There are some surprise allocations in here because
//! doing the conversion without transmute requires two steps.
//! It also requires a nightly feature...

extern crate byte_tools;
pub use byte_tools::*;

// TODO missing: big endian
// TODO missing: f32
// TODO missing: read_f64_le/write_f64_le (non-slice versions)

pub fn read_f64v_le(dst: &mut [f64], src: &[u8]) {
    let mut u64s = vec![0u64; dst.len()];

    read_u64v_le(&mut u64s, src);

    for (f, i) in dst.iter_mut().zip(u64s) {
        *f = f64::from_bits(i);
    }
}

pub fn write_f64v_le(dst: &mut [u8], src: &[f64]) {
    let mut u64s = vec![0u64; src.len()];

    for (i, &f) in u64s.iter_mut().zip(src) {
        *i = f.to_bits();
    }

    write_u64v_le(dst, &u64s);
}
