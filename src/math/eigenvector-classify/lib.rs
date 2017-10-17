
#[macro_use] extern crate itertools;
extern crate slice_of_array;

use ::std::iter::FromIterator;
use ::std::collections::{HashMap, HashSet};
use ::std::hash::Hash;

use ::slice_of_array::prelude::*;

pub fn acousticness(ev: &[[f64; 3]], mask: &[u32; 3]) -> f64
{ keyed_acousticness(ev, &vec![Some(()); ev.len()], mask) }

pub fn keyed_acousticness<K: Eq + Hash>(ev: &[[f64; 3]], keys: &[Option<K>], mask: &[u32; 3]) -> f64
{ keyed_acoustic_basis(keys, mask).probability(ev) }

/// More efficient than 'keyed_acousticness' if you need the same basis multiple times.
///
/// Use the 'probability' method on the returned object.
pub fn keyed_acoustic_basis<K: Eq + Hash>(keys: &[Option<K>], mask: &[u32; 3]) -> Basis<[f64; 3]>
{ keyed_basis(keys).as_ref().on_axes(mask) }

pub fn polarization(ev: &[[f64; 3]]) -> [f64; 3]
{
    // inefficient HACK
    // really just normed sum along each of three components
    let all_unique: Vec<_> = (0..ev.len()).map(Some).collect();
    let x = keyed_acousticness(ev, &all_unique, &[1, 0, 0]);
    let y = keyed_acousticness(ev, &all_unique, &[0, 1, 0]);
    let z = keyed_acousticness(ev, &all_unique, &[0, 0, 1]);
    [x, y, z]
}

#[derive(Debug, Clone)]
pub struct Basis<T> {
    dim: usize,
    data: Vec<T>,
}

// this is not public so that I don't have to worry
// about feature parity between ownership variants
#[derive(Debug, Copy, Clone)]
struct Slice<'a, T: 'a> {
    dim: usize,
    data: &'a [T],
}

impl<T> Basis<T> {
    fn as_ref(&self) -> Slice<T> {
        let Basis { dim, ref data } = *self;
        Slice { dim, data }
    }
}

impl<'a> Slice<'a, f64>
{
    fn on_axes(&self, mask: &[u32; 3]) -> Basis<[f64; 3]> {
        assert!(mask.iter().all(|&x| x == 0 || x == 1));

        let axes: Vec<_> =
            mask.iter().enumerate()
            .filter(|&(_, &x)| x == 1)
            .map(|(i, _)| i)
            .collect();

        let mut out = vec![[0.0; 3]; self.data.len() * axes.len()];
        {
            let mut out_kets = out.chunks_mut(self.dim);
            for k in axes {
                for ket in self.kets() {
                    let out_ket = out_kets.next().unwrap();
                    for (a, &b) in izip!(out_ket, ket) {
                        a[k] = b;
                    }
                }
            }
        }
        Basis { data: out, dim: self.dim }
    }

    fn kets(&self) -> ::std::slice::Chunks<f64> { self.data.chunks(self.dim) }

    fn probability(&self, ket: &[f64]) -> f64
    {
        let square = |x| x * x;
        let dot = |a: &[f64], b: &[f64]| {
            assert_eq!(a.len(), b.len());
            izip!(a, b).map(|(a, b)| a * b).sum()
        };
        let sqnorm: f64 = dot(ket, ket);

        // FIXME square(dot()) or just dot()? WHY AM I NOT SURE?!
        self.kets().map(|bra| square(dot(bra, ket))).sum::<f64>() / sqnorm
    }
}

impl Basis<[f64; 3]>
{
    fn flat(&self) -> Slice<f64> {
        Slice {
            dim: self.dim * 3,
            data: self.data.flat(),
        }
    }

    /// Get the total fraction from 0.0 to 1.0 of the ket
    /// that is included in this Basis.
    pub fn probability(&self, ket: &[[f64; 3]]) -> f64 { self.flat().probability(ket.flat()) }
}

fn keyed_basis<K: Eq + Hash>(keys: &[Option<K>]) -> Basis<f64>
{
    // note: map key type is actually &K
    let mut by_key = HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        if let Some(ref key) = *key {
            by_key.entry(key)
                .or_insert_with(Vec::new)
                .push(i);
        }
    }

    let mut data = by_key.values().map(|indices| {
        let indices = HashSet::<usize>::from_iter(indices.iter().cloned());
        (0..keys.len()).map(move |i| indices.contains(&i) as i32 as f64)
    }).flat_map(|x| x).collect::<Vec<_>>();

    for ket in data.chunks_mut(keys.len()) {
        let rnorm = ket.iter().map(|x| x*x).sum::<f64>().sqrt().recip();
        for x in ket { *x *= rnorm; }
    }

    Basis { dim: keys.len(), data }
}

