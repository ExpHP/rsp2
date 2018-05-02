use ::{Structure, Lattice, CoordsKind};
use ::{Perm};

use ::rsp2_array_utils::{arr_from_fn, try_arr_from_fn};

use ::rsp2_array_types::{V3};


// ---------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Builder {
    // supercell "matrix". (currently only diagonal are supported)
    diagonal: [u32; 3],
    // add a lattice point to the entire output.  This will be reflected
    // in `supercell_indices()` as well.
    offset: V3<i32>,
}

pub fn diagonal(dims: [u32; 3]) -> Builder {
    // there's really no use case for empty supercells AFAICT, and they break
    // the laws of periodicity
    assert!(dims.iter().all(|&x| x > 0), "supercell of size zero?!");
    Builder {
        diagonal: dims,
        offset: V3([0; 3]),
    }
}

/// Given dims `[a, b, c]`, makes a supercell of size `[2a + 1, 2b + 1, 2c + 1]`.
pub fn centered_diagonal(extra_images: [u32; 3]) -> Builder {
    let extra_images = V3(extra_images);

    Builder {
        diagonal: (extra_images * 2 + V3([1; 3])).0,
        offset: extra_images.map(|x| -(x as i32)),
    }
}

impl Builder {
    pub fn build<M>(&self, structure: Structure<M>) -> (Structure<M>, SupercellToken)
    where M: Clone,
    {
        self.build_with(structure, |meta, _| meta.clone())
    }

    pub fn build_with<M, M2, F>(&self, structure: Structure<M>, make_meta: F) -> (Structure<M2>, SupercellToken)
    where F: FnMut(&M, [u32; 3]) -> M2,
    {
        diagonal_with(self.clone(), structure, make_meta)
    }
}

// ---------------------------------------------------------------

impl Builder {
    // convert into a data structure with precomputed info for general supercell matrices
    fn into_sc_token(self, num_primitive_atoms: usize) -> SupercellToken {
        let Builder { offset, diagonal: periods } = self;
        let integer_lattice = Lattice::diagonal(&V3(periods).map(|x| x as f64));
        SupercellToken { offset, periods, integer_lattice, num_primitive_atoms }
    }
}

// ---------------------------------------------------------------

fn diagonal_with<M, M2, F>(builder: Builder, structure: Structure<M>, mut make_meta: F)
-> (Structure<M2>, SupercellToken)
where F: FnMut(&M, [u32; 3]) -> M2,
{
    let Structure { lattice, coords, meta } = structure;

    // construct a SupercellToken ASAP so that we know the rest of the code
    // works for general supercell matrices
    let sc = builder.into_sc_token(coords.len());

    // number of offsets along each lattice vector.
    let num_sc = (sc.periods[0] * sc.periods[1] * sc.periods[2]) as usize;
    let num_supercell_atoms = num_sc * coords.len();

    let sc_carts = sc_lattice_vecs(sc.periods, sc.offset, &lattice);
    let mut new_carts = Vec::with_capacity(num_supercell_atoms);
    for atom_cart in coords.into_carts(&lattice) {
        let old_len = new_carts.len();
        new_carts.extend_from_slice(&sc_carts);
        ::util::translate_mut_n3_3(&mut new_carts[old_len..], &atom_cart);
    }

    let sc_idx = sc_indices(sc.periods);
    let mut new_meta = Vec::with_capacity(num_supercell_atoms);
    for m in meta {
        for idx in &sc_idx {
            new_meta.push(make_meta(&m, idx.0));
        }
    }

    let structure = Structure {
        lattice: &sc.integer_lattice * &lattice,
        coords: CoordsKind::Carts(new_carts),
        meta: new_meta,
    };

    (structure, sc)
}

/// Contains enough information to deconstruct a supercell produced by this library.
///
/// The order of the atoms in the supercell is unspecified and may change; you should
/// not assume that the images are grouped by unit cell or by primitive site.
/// Instead, please refer to the methods `cell_indices` and `primitive_site_indices`.
pub struct SupercellToken {
    periods: [u32; 3],
    offset: V3<i32>,
    num_primitive_atoms: usize,
    // supercell in units of primitive cell vectors.  Elements are integral
    integer_lattice: Lattice,
}

#[derive(Debug, Fail)]
#[fail(display = "Suspiciously large movement between supercell images: {:e}", magnitude)]
pub struct BigDisplacement {
    backtrace: ::failure::Backtrace,
    magnitude: f64,
}

pub type OwnedMetas<'a,T> = ::std::vec::Drain<'a,T>;
impl SupercellToken {

    /// The number of images taken of the unit cell contained in the supercell.
    #[inline]
    pub fn num_cells(&self) -> usize {
        (self.periods[0] * self.periods[1] * self.periods[2]) as usize
    }

    #[inline]
    pub fn num_primitive_atoms(&self) -> usize {
        self.num_primitive_atoms
    }

    #[inline]
    pub fn num_supercell_atoms(&self) -> usize {
        self.num_cells() * self.num_primitive_atoms()
    }

    /// Takes data for each atom of the primitive cell and expands it to the
    /// size of the supercell.
    #[inline]
    pub fn replicate<M>(&self, vec: &[M]) -> Vec<M>
    where M: Clone
    {
        let mut out = Vec::with_capacity(vec.len() * self.num_cells());
        for m in vec {
            let new_len = out.len() + self.num_cells();
            out.resize(new_len, m.clone());
        }
        out
    }

    /// Recover a primitive cell by averaging positions from a supercell.
    ///
    /// Uses metadata from the first image of each atom.
    ///
    /// May fail if any of the following have occurred since the token was created:
    /// * Addition or deletion of atoms
    /// * Reordering of atoms
    /// * Wrapping of positions (FIXME unnecessary limitation)
    /// * Images of an atom did not move by equal amounts (within `validation_radius`)
    #[inline]
    pub fn deconstruct<M>(&self, validation_radius: f64, structure: Structure<M>)
    -> Result<Structure<M>, BigDisplacement>
    {
        self.deconstruct_with(
            validation_radius,
            structure,
            |mut metas| metas.next().unwrap())
    }

    /// Recover a primitive cell by averaging positions from a supercell.
    ///
    /// `fold_meta` is called on the images of each primitive atom.
    /// The primitive atoms are done in an arbitrary order.
    ///
    /// May fail if any of the following have occurred since the token was created:
    /// * Addition or deletion of atoms
    /// * Reordering of atoms
    /// * Wrapping of positions (FIXME unnecessary limitation)
    /// * Images of an atom did not move by equal amounts (within `validation_radius`)
    pub fn deconstruct_with<M, M2, F>(
        &self,
        validation_radius: f64,
        structure: Structure<M2>,
        mut fold_meta: F,
    ) -> Result<Structure<M>, BigDisplacement>
    where F: FnMut(OwnedMetas<M2>) -> M,
    {
        assert_eq!(
            structure.num_atoms(), self.num_supercell_atoms(),
            "wrong # of atoms in supercell",
        );

        let num_cells = self.num_cells();
        let SupercellToken { periods, offset, ref integer_lattice, num_primitive_atoms } = *self;
        let Structure { lattice, coords, meta } = structure;

        let primitive_lattice = integer_lattice.inverse_matrix() * &lattice;

        let out_carts = {
            let neg_offsets = {
                let mut vs = sc_lattice_vecs(periods, offset, &primitive_lattice);
                for v in &mut vs {
                    *v *= -1.0;
                }
                vs
            };

            let mut carts = coords.into_carts(&lattice);
            let mut image_carts = Vec::with_capacity(num_cells);
            let mut out_carts = Vec::with_capacity(num_primitive_atoms);
            while !carts.is_empty() {
                // Fold all images of a single atom
                let new_len = carts.len() - num_cells;

                // Map into the primitive cell by subtracting the same lattice vectors
                // that were used to produce the images.
                //
                // The images are very deliberately NOT wrapped using modulo logic,
                // because that would interfere with our ability to average the positions
                // of an atom near the periodic boundary.
                image_carts.clear();
                image_carts.extend(carts.drain(new_len..));
                ::util::translate_mut_n3_n3(&mut image_carts, &neg_offsets);

                out_carts.push(V3(try_arr_from_fn(|k| {
                    let this_axis = || image_carts.iter().map(|v| v[k]);

                    let inf = ::std::f64::INFINITY;
                    let min = this_axis().fold(inf, |a, b| a.min(b));
                    let max = this_axis().fold(-inf, |a, b| a.max(b));
                    if max - min > 2.0 * validation_radius {
                        let backtrace = ::failure::Backtrace::new();
                        let magnitude = max - min;
                        return Err(BigDisplacement { backtrace, magnitude });
                    }

                    let sum = this_axis().sum::<f64>();
                    Ok(sum / num_cells as f64)
                })?));
            }
            // Atoms were done in reverse order
            out_carts.into_iter().rev().collect()
        };

        let out_meta = {
            let mut meta = meta;
            let mut out_meta = Vec::with_capacity(num_primitive_atoms);
            while !meta.is_empty() {
                // Fold all images of a single atom
                let new_len = meta.len() - num_cells;
                out_meta.push(fold_meta(meta.drain(new_len..)));
            }
            // Atoms were done in reverse order
            out_meta.into_iter().rev().collect()
        };

        Ok(Structure {
            lattice: primitive_lattice,
            coords: CoordsKind::Carts(out_carts),
            meta: out_meta,
        })
    }

    /// Defines which image of the primitive cell each atom in the supercell belongs to.
    ///
    /// Each image of the primitive cell is represented by the integral coordinates of the
    /// lattice vector used to produce it. It is guaranteed that the output will contain
    /// `self.num_cells()` unique values, and that each one will appear exactly
    /// `self.num_primitive_atoms()` times.
    ///
    /// This function is constant for any given supercell; it merely exists to define the
    /// conventions for ordering used by the library.
    pub fn cell_indices(&self) -> Vec<V3<u32>> {
        sc_indices(self.periods).iter().cloned()
            .cycle().take(self.num_supercell_atoms())
            .collect()
    }

    /// Gives coefficients of the lattice vectors that were added to each atom in the supercell.
    ///
    /// This is like `cell_indices`, but gives the indices as signed integer offsets.
    /// The `offset` from building the supercell affects this, so that e.g. centered supercells
    /// have a signed index of `[0, 0, 0]` assigned to atoms in the centermost unit cell.
    ///
    /// This function is constant for any given supercell; it merely exists to define the
    /// conventions for ordering used by the library.
    pub fn signed_cell_indices(&self) -> Vec<V3<i32>> {
        signed_sc_indices(self.periods, self.offset).iter().cloned()
            .cycle().take(self.num_supercell_atoms())
            .collect()
    }

    /// Defines which atom in the primitive cell that each supercell atom is an image of.
    ///
    /// This function is constant for any given supercell; it merely exists to define the
    /// conventions for ordering used by the library.
    pub fn primitive_site_indices(&self) -> Vec<usize> {
        self.replicate(&(0..self.num_primitive_atoms).collect::<Vec<_>>())
    }

    /// Get a depermutation representing translation by a unit cell lattice point.
    ///
    /// Please see `conventions.md` for an explanation of depermutations.
    pub fn lattice_point_translation_deperm(&self, index: V3<i32>) -> Perm {
        // Depermutations that permute the cells along each axis independently.
        // (expressed in the quotient space of images along that axis)
        let axis_deperms: [_; 3] = arr_from_fn(|k| {
            Perm::eye(self.periods[k]).shift_signed(index[k])
        });

        // Construct the overall deperm as an outer product of deperms.
        // this could be written as a fold, but I wanted to emphasize the order;
        // the innermost perm must correspond to the fastest-varying index.
        Perm::eye(self.num_primitive_atoms as u32)
            .with_inner(&axis_deperms[0])
            .with_inner(&axis_deperms[1])
            .with_inner(&axis_deperms[2])
    }
}

// supercell indices in the library's preferred order
fn sc_indices(periods: [u32; 3]) -> Vec<V3<u32>> {
    let mut out = Vec::with_capacity((periods[0] * periods[1] * periods[2]) as usize);
    for ia in 0..periods[0] {
        for ib in 0..periods[1] {
            for ic in 0..periods[2] {
                out.push(V3([ia, ib, ic]));
            }
        }
    }
    out
}

// signed lattice point indices in the library's preferred order
fn signed_sc_indices(periods: [u32; 3], offset: V3<i32>) -> Vec<V3<i32>> {
    sc_indices(periods).into_iter()
        .map(|idx| idx.map(|x| x as i32) + offset)
        .collect()
}

// cartesian supercell image offsets in the library's preferred order
fn sc_lattice_vecs(periods: [u32; 3], offset: V3<i32>, lattice: &Lattice) -> Vec<V3> {
    signed_sc_indices(periods, offset).into_iter()
        .map(|idx| idx.map(|x| x as f64) * lattice.matrix())
        .collect()
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use ::{Permute, Perm};
    use ::{CoordsKind, Structure, Lattice};
    use ::rsp2_array_types::{V3, Envee};

    #[test]
    fn diagonal_supercell_smoke_test() {
        let coords = CoordsKind::Fracs(vec![[0.0, 0.0, 0.0]].envee());

        let original = Structure::new_coords(Lattice::eye(), coords);
        let (supercell, sc_token) = ::supercell::diagonal([2, 2, 2]).build(original.clone());

        assert_eq!(supercell.num_atoms(), 8);
        assert_eq!(supercell.lattice(), &Lattice::cubic(2.0));

        assert!(::util::eq_unordered_n3(&supercell.to_carts(), [
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
        ].envee_ref()));

        let deconstructed = sc_token.deconstruct(1e-10, supercell).unwrap();

        assert_eq!(original.to_carts(), deconstructed.to_carts());
        assert_eq!(original.lattice(), deconstructed.lattice());
    }

    #[test]
    fn test_diagonal_supercell() {
        // nondiagonal lattice so that matrix multiplication order matters.
        // carefully chosen so that the inverse has an exact representation.
        let lattice = Lattice::from(&[
            [2.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);

        let coords = CoordsKind::Fracs(vec![
            [ 0.5, -0.5, 0.0], // cart: [+1.0, -1.0,  0.0]
            [ 0.0,  0.5, 0.5], // cart: [ 0.0, +1.0, +1.0]
        ].envee());

        let original = Structure::new_coords(lattice, coords);
        let (supercell, sc_token) = ::supercell::diagonal([4, 2, 2]).build(original.clone());
        let deconstructed = sc_token.deconstruct(1e-10, supercell.clone()).unwrap();

        assert_eq!(original.to_carts(), deconstructed.to_carts());
        assert_eq!(original.lattice(), deconstructed.lattice());

        // test error on unequal shifts
        let mut supercell = supercell;
        let mut carts = supercell.to_carts();
        carts[4][1] += 1e-6;
        supercell.coords = CoordsKind::Carts(carts);
        assert!(sc_token.deconstruct(1e-10, supercell.clone()).is_err());
    }

    #[test]
    fn test_centered_diagonal_supercell() {
        // nondiagonal lattice so that matrix multiplication order matters
        // carefully chosen so that the inverse has an exact representation.
        let lattice = Lattice::from(&[
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 8.0],
        ]);

        let coords = CoordsKind::Carts(vec![
            [0.25, 0.75, 1.5],
        ].envee());

        let original = Structure::new_coords(lattice, coords);
        let (supercell, sc_token) = {
            ::supercell::centered_diagonal([0, 2, 1])
                .build(original.clone())
        };
        let deconstructed = sc_token.deconstruct(1e-10, supercell.clone()).unwrap();

        assert_eq!(original.to_carts(), deconstructed.to_carts());
        assert_eq!(original.lattice(), deconstructed.lattice());
        let expected_carts = {
            let xs = [0.25];
            let ys = [-7.25, -3.25, 0.75, 4.75, 8.75];
            let zs = [-6.5, 1.5, 9.5];
            iproduct!(&xs, &ys, &zs)
                .map(|(&x, &y, &z)| [x, y, z])
                .collect::<Vec<_>>().envee()
        };
        let actual_carts = supercell.to_carts();
        assert!(::util::eq_unordered_n3(&expected_carts, &actual_carts), "{:?} {:?}", expected_carts, actual_carts);
    }

    #[test]
    fn lattice_point_deperms() {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
        struct Label {
            atom: u32,
            cell: [u32; 3],
        }

        // make a superstructure whose metadata uniquely labels the sites.
        let coords = CoordsKind::Carts(vec![
            // something with an exact representation
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ].envee());
        let lattice = Lattice::eye();
        let structure = Structure::new(lattice, coords, vec![0, 1]);
        let (superstructure, sc_token) = {
            // supercell is chosen to give exact floating representation.
            // some dimensions are larger than 2 to ensure that there exist translational
            //  symmetries that are not equal to their own inverses.
            ::supercell::diagonal([8, 2, 4])
                .build_with(structure, |&atom, cell| Label { atom, cell })
        };

        for _ in 0..10 {
            let lattice_point = {
                use ::rand::{thread_rng, Rng};
                V3::from_fn(|_| thread_rng().gen_range(-15, 15))
            };

            // translating the superstructure by a primitive cell lattice point...
            let translated = {
                let mut s = superstructure.clone();
                // (this is deliberately cartesian because we want to use a lattice point
                //  of the primitive lattice, not the superlattice.
                //  This works because the primitive lattice was set to be the identity matrix)
                s.translate_cart(&lattice_point.map(|x| x as f64));
                s
            };

            // ...should be equivalent to applying the depermutation to the metadata...
            let depermuted = {
                let mut s = superstructure.clone();
                let deperm = sc_token.lattice_point_translation_deperm(lattice_point);
                let permuted_meta = s.metadata().to_vec().permuted_by(&deperm);
                s.set_metadata(permuted_meta);
                s
            };

            // ...when comparing positions for the same label, reduced into the supercell.
            let canonicalized_coords = |mut structure: Structure<_>| {
                let perm = Perm::argsort(structure.metadata());
                structure = structure.permuted_by(&perm);
                structure.reduce_positions();
                structure.to_carts()
            };
            assert_eq!(
                canonicalized_coords(translated),
                canonicalized_coords(depermuted),
            );
        }
    }
}
