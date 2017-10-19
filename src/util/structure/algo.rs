pub(crate) mod layer {
    use ::{Structure, Lattice};

    use ::std::mem;
    use ::itertools::Itertools;
    use ::ordered_float::NotNaN;

    /// Newtype wrapper for a layer number.
    ///
    /// They are numbered from 0.
    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Hash, Eq, Ord)]
    pub struct Layer(pub u32);

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
    }
}

pub(crate) mod perm {
    use ::slice_of_array::prelude::*;
    use ::{Lattice, CoordStructure};
    use ::symmops::FracRot;
    error_chain!{ }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Perm(Vec<u32>);

    pub fn of_rotation(
        structure: &CoordStructure,
        rotation: &FracRot,
        tol: f64,
    ) -> Result<Perm>
    {Ok({
        let lattice = structure.lattice();
        let from_fracs = structure.to_fracs();
        let to_fracs = rotation.transform(&from_fracs);

        of_rotation_impl(lattice, &from_fracs, &to_fracs, tol)?
    })}

    fn of_rotation_impl(
        lattice: &Lattice,
        from_fracs: &[[f64; 3]],
        to_fracs: &[[f64; 3]],
        tol: f64,
    ) -> Result<Perm>
    {Ok({
        use ::rsp2_array_utils::dot;
        use ::ordered_float::NotNaN;
        use ::Coords::Fracs;

        // Sort both sides by some measure which is likely to produce a small
        // maximum value of (sorted_rotated_index - sorted_original_index).
        // The C code is optimized for this case, reducing an O(n^2)
        // search down to ~O(n). (for O(n log n) work overall, including the sort)
        //
        // We choose distance from the nearest bravais lattice point as our measure.
        let sort_by_lattice_distance = |fracs: &[[f64; 3]]| {
            let mut fracs = fracs.to_vec();
            for x in fracs.flat_mut() {
                *x -= x.round();
            }

            let distances = Fracs(fracs.clone())
                    .to_carts(lattice)
                    .into_iter()
                    .map(|x| NotNaN::new(dot(&x, &x).sqrt()).unwrap())
                    .collect::<Vec<_>>();
            let perm = argsort(&distances);
            (perm.clone(), permute(&fracs, &perm))
        };

        let (perm_from, sorted_from) = sort_by_lattice_distance(&from_fracs);
        let (perm_to, sorted_to) = sort_by_lattice_distance(&to_fracs);

        let perm_between = brute_force_near_identity(
            lattice,
            &sorted_from[..],
            &sorted_to[..],
            tol,
        )?;

        // Compose all of the permutations for the full permutation.
        //
        // Note the following properties of permutation arrays:
        //
        // 1. Inverse:         if  x[perm] == y  then  x == y[argsort(perm)]
        // 2. Associativity:   x[p][q] == x[p[q]]
        let perm = perm_from.0;
        let perm = permute(&perm, &perm_between);
        let perm = permute(&perm, &invert(&perm_to));
        Perm(perm)
    })}

    fn permute<T: Clone>(xs: &[T], perm: &Perm) -> Vec<T>
    {
        assert_eq!(xs.len(), perm.0.len());
        perm.0.iter().map(|&i| xs[i as usize].clone()).collect()
    }

    fn sort<T: Ord + Clone>(xs: &[T]) -> (Perm, Vec<T>)
    {
        let mut xs: Vec<_> = xs.iter().cloned().enumerate().collect();
        xs.sort_by(|a, b| a.1.cmp(&b.1));
        let (perm, xs) = xs.into_iter().map(|(i, x)| (i as u32, x)).unzip();
        (Perm(perm), xs)
    }

    fn argsort<T: Ord + Clone>(xs: &[T]) -> Perm
    { sort(xs).0 }

    fn invert(perm: &Perm) -> Perm
    {
        // bah. less code to test...
        argsort(&perm.0)
    }

    // Optimized for permutations near the identity.
    fn brute_force_near_identity(
        lattice: &Lattice,
        from_fracs: &[[f64; 3]],
        to_fracs: &[[f64; 3]],
        tol: f64,
    ) -> Result<Perm>
    {Ok({

        assert_eq!(from_fracs.len(), to_fracs.len());
        let n = from_fracs.len();

        const UNSET: u32 = ::std::u32::MAX;
        assert!(n < UNSET as usize);

        let mut perm = vec![UNSET; from_fracs.len()];

        // optimization: Rather than filling the out vector in order,
        // we find where each index belongs (e.g. we place the 0, then
        // we place the 1, etc.).
        // Then we can track the first unassigned index.
        //
        // This works best if the permutation is close to the identity.
        // (more specifically, if the max value of 'out[i] - i' is small)
        //
        // This optimization does create some data dependencies which block
        // opportunities for parallelization; but for reducing O(n^2)
        // computations down to O(n), it is worth it.
        let mut search_start = 0;

        'from: for from in 0..n {

            // Skip through things filled out of order.
            while perm[search_start] != UNSET {
                search_start += 1;
            }

            for to in search_start..n {
                if perm[to] != UNSET {
                    continue;
                }

                // FIXME use utils
                let mut diff = [0f64; 3];
                for k in 0..3 {
                    diff[k] = from_fracs[from][k] - to_fracs[to][k];
                    diff[k] -= diff[k].round();
                }
                let mut distance2 = 0.0;
                for k in 0..3 {
                    let mut diff_cart = 0.0;
                    for l in 0..3 {
                        diff_cart += lattice.matrix()[k][l] * diff[l];
                    }
                    distance2 += diff_cart * diff_cart;
                }

                if distance2 < tol * tol {
                    perm[to] = from as u32;
                    continue 'from;
                }
            }
            bail!("positions are too dissimilar");
        }

        ensure!(
            perm.iter().all(|&x| x != UNSET),
            "multiple positions mapped to the same index");

        Perm(perm)
    })}

    #[cfg(test)]
    mod tests {
        use ::Lattice;
        use super::Perm;

        use ::slice_of_array::*;
        use ::rand::{Rng, Rand};

        // hmmm, this is just begging for quickcheck...
        fn random_perm(n: u32) -> Perm
        {
            let mut perm: Vec<_> = (0..n as u32).collect();
            ::rand::thread_rng().shuffle(&mut perm);
            Perm(perm)
        }

        fn random_vec<T: Rand>(n: u32) -> Vec<T>
        { (0..n).map(|_| ::rand::random()).collect() }

        fn random_problem(n: u32) -> (Vec<[f64; 3]>, Perm, Vec<[f64; 3]>)
        {
            let original: Vec<[f64; 3]> = random_vec(n);
            let perm = random_perm(n);
            let permuted = super::permute(&original, &perm);
            (original, perm, permuted)
        }

        #[test]
        fn perm_inverse()
        {
            const SIZE: u32 = 20;
            let perm = random_perm(SIZE);
            let inv = super::invert(&perm);

            assert!(super::permute(&perm.0, &inv).into_iter().eq(0..SIZE));
            assert!(super::permute(&inv.0, &perm).into_iter().eq(0..SIZE));
        }

        #[test]
        fn brute_force_works() {
            let (original, perm, permuted) = random_problem(20);
            let lattice = Lattice::new(random_vec(3).as_array());

            let output = super::brute_force_near_identity(
                &lattice, &original, &permuted, 1e-5,
            ).unwrap();

            assert_eq!(output, perm);
        }

        #[test]
        fn of_rotation_impl_works() {
            let (original, perm, permuted) = random_problem(20);
            let lattice = Lattice::new(random_vec(3).as_array());

            let output = super::of_rotation_impl(
                &lattice, &original, &permuted, 1e-5,
            ).unwrap();

            assert_eq!(output, perm);
        }
    }
}
