use ::Result;
use ::{Structure, Lattice};

use ::std::mem;
use ::itertools::Itertools;
use ::ordered_float::NotNaN;

/// Newtype wrapper for a layer number.
///
/// They are numbered from 0.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Hash, Eq, Ord)]
pub struct Layer(pub u32);

// FIXME this is wrong wrong wrong.
//       Only correct when the other two lattice vectors are
//         perpendicular to the chosen lattice vector.
//       May need to rethink the API.
//
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
-> Result<(Vec<Layer>, u32)>
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
-> Result<(Vec<Layer>, u32)>
{Ok({
    if fracs.len() == 0 {
        return Ok((vec![], 0));
    }

    let axis = {
        let mut sorted = *normal;
        sorted.sort_unstable();
        ensure!(sorted == [0, 0, 1],
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

    // FIXME: On second thought I think this is incorrect.
    //        Our requirement should not be that the normal is a
    //        lattice vector; but rather, that two of the lattice
    //        vectors lie within the plane.

    { // Safety HACK!
        use ::rsp2_array_utils::dot;
        let lengths = lattice.lengths();
        let vecs = lattice.matrix();
        for k in 0..3 {
            if k != axis {
                let cos = dot(&vecs[k], &vecs[axis]) / (lengths[k] * lengths[axis]);
                ensure!(cos.abs() < 1e-7,
                    "For your safety, assign_layers is currently limited to \
                    lattices where the normal is perpendicular to the other two \
                    lattice vectors.");
            }
        }
    }

    // --(original (incorrect) text)--
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
    // --(end original text)--

    // Split the positions into contiguous segments of atoms
    // where the distance between any two consecutive atoms
    // (projected onto the normal vector) is at most `sep`.
    let mut groups = vec![];
    let mut cur_group = vec![sorted[0].0];
    for (&(_, ax), &(bi, bx)) in sorted.iter().tuple_windows() {
        // Invariant
        assert!(cur_group.len() >= 1);

        if bx - ax > frac_sep {
            let done = mem::replace(&mut cur_group, vec![]);
            groups.push(done);
        }
        cur_group.push(bi);
    }

    // Detect a layer crossing the periodic plane:
    // If the first and last "layers" are close enough,
    // join them together.
    let wrap_distance = sorted[0].1 - (sorted.last().unwrap().1 - 1.0);
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
})}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use ::{Permute, Perm};

    #[test]
    fn assign_layers_impl() {
        let go = super::assign_layers_impl;

        let fracs = vec![
            // we will be looking along y with frac_tol = 0.11
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0], // obviously same layer
            [0.8, 0.3, 0.4], // laterally displaced, but still same layer
            [0.0, 0.7, 0.0], // a second layer
            [0.0, 0.8, 0.0],
            // (first and last not close enough to meet)
        ];

        // NOTE this does try a non-diagonal lattice and even
        //      goes along an awkwardly oriented vector
        //       but we restrict it to a form that the
        //       (currently broken) algorithm will work on
        //      (1st and 3rd vecs must be orthogonal to 2nd vec)
        let ylen = 4.0;
        let cart_tol    = 0.11 * ylen;  // produces 2 layers
        let smaller_tol = 0.09 * ylen;  // makes all atoms look separate

        const IR2: f64 = ::std::f64::consts::FRAC_1_SQRT_2;
        let lattice = Lattice::new(&[
            [ ylen * IR2, ylen *  IR2,  0.0],
            [ ylen * IR2, ylen * -IR2,  0.0], // (axis we're using)
            [        0.0,         0.0, 13.0],
        ]);

        let layers = |xs: Vec<_>| xs.into_iter().map(Layer).collect();

        assert_eq!(
            go(&fracs, &lattice, &[0, 1, 0], cart_tol).unwrap(),
            (layers(vec![0, 0, 0, 1, 1]), 2));

        // put them out of order
        let (fracs, perm) = ::oper::perm::shuffle(&fracs);
        assert_eq!(
            go(&fracs, &lattice, &[0, 1, 0], cart_tol).unwrap().0,
            layers(vec![0, 0, 0, 1, 1]).permuted_by(&perm));

        // try a smaller tolerance
        assert_eq!(
            go(&fracs, &lattice, &[0, 1, 0], smaller_tol).unwrap().0,
            layers(vec![0, 1, 2, 3, 4]).permuted_by(&perm));

        // try bridging across the periodic boundary to
        //   join the two layers.
        // also, try a position outside the unit cell.
        let (mut fracs, mut perm) = (fracs, perm);
        fracs.push([0.0, 1.9, 0.0]);
        fracs.push([0.0, 0.0, 0.0]);
        perm.append_mut(&Perm::eye(2));

        assert_eq!(
            go(&fracs, &lattice, &[0, 1, 0], cart_tol).unwrap().0,
            layers(vec![0, 0, 0, 0, 0, 0, 0]).permuted_by(&perm));

        // try joining the end regions when there is more than one layer
        // (this might use a different codepath for some implementations)
        fracs.push([0.0, 0.5, 0.0]);
        perm.append_mut(&Perm::eye(1));

        assert_eq!(
            go(&fracs, &lattice, &[0, 1, 0], cart_tol).unwrap().0,
            layers(vec![0, 0, 0, 0, 0, 0, 0, 1]).permuted_by(&perm));
    }
}
