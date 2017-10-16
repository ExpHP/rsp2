use ::{Structure, Lattice};

use ::std::mem;
use ::itertools::Itertools;
use ::ordered_float::NotNaN;

/// Determine layers in a structure, numbered from zero.
/// Also returns the count.
///
/// This finds "layers" defined as groups of atoms isolated from
/// other groups by at least some minimum distance projected along
/// a normal vector.
///
/// Normal is in fractional coords, and is currently limited such
/// that it must be one of the lattice vectors.
pub fn assign_layers<M>(structure: &Structure<M>, normal: &[i32; 3], sep: f64)
-> (Vec<Layer>, u32)
{
    assign_layers_impl(
        &structure.to_fracs(),
        structure.lattice(),
        normal,
        sep,
    )
}

// monomorphic for less codegen
fn assign_layers_impl(fracs: &[[f64; 3]], lattice: &Lattice, normal: &[i32; 3], sep: f64)
-> (Vec<Layer>, u32)
{
    if fracs.len() == 0 {
        return (vec![], 0);
    }

    let axis = {
        let mut sorted = *normal;
        sorted.sort_unstable();
        assert!(sorted == [0, 0, 1],
            "unsupported layer normal: {:?}", normal);

        normal.iter().position(|&x| x == 1).unwrap()
    };
    let reduce = |x: f64| (x.fract() + 1.0).fract();

    let sorted: Vec<(usize, f64)> = {
        let mut vec: Vec<_> = fracs.iter()
            .map(|f| reduce(f[axis]))
            .enumerate().collect();

        vec.sort_by_key(|&(_, x)| NotNaN::new(x).unwrap());
        vec
    };

    let frac_sep = sep / lattice.lengths()[axis];

    // NOTE: the validity of the following algorithm is
    //       predicated on the normal pointing precisely along
    //       a lattice vector.  This ensures that there's no
    //       funny business where the projected distance along the
    //       axis could suddenly change as a particle crosses a
    //       periodic surface while traveling within a layer.
    //
    //       Some other directions with integer coordinates
    //       could be handled in the future by a unimodular
    //       transform to make that direction become one of the
    //       lattice vectors....In theory.

    // Split the positions into contiguous segments of atoms
    // where the distance between any two consecutive atoms
    // (projected onto the normal vector) is at most `sep`.
    let mut groups = vec![];
    let mut cur_group = vec![sorted[0].0];
    for (&(_, ax), &(bi, bx)) in sorted.iter().tuple_windows() {
        // Invariant
        assert!(cur_group.len() >= 1);

        if bx - ax <= frac_sep {
            let done = mem::replace(&mut cur_group, vec![]);
            groups.push(done);
        }
        cur_group.push(bi);
    }

    // Detect a layer crossing the periodic plane:
    // If the first and last "layers" are close enough,
    // join them together.
    let wrap_distance = sorted[0].1 - (1.0 - sorted.last().unwrap().1);
    if wrap_distance <= frac_sep && !groups.is_empty() {
        groups[0].extend(cur_group); // join them
    } else {
        groups.push(cur_group); // keep them separate
    }

    // go from vecs of indices to vec of layers
    let n_layers = groups.len();
    let mut out = vec![Layer(0xf00d); fracs.len()];
    for (layer, group) in groups.into_iter().enumerate() {
        for i in group {
            out[i] = Layer(layer as u32);
        }
    }
    (out, n_layers as u32)
}


/// Newtype wrapper for a layer number.
///
/// They are numbered from 0.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Hash, Eq, Ord)]
pub struct Layer(pub u32);
