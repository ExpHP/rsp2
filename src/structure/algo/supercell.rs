use ::{Structure, Lattice, CoordsKind, Coords};

use ::rsp2_soa_ops::{Perm};
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
    pub fn absolute_offset(mut self, min_image: V3<i32>) -> Builder {
        self.offset = min_image; self
    }

    pub fn center(mut self, center: V3<i32>) -> Builder {
        let natural_center = V3(self.diagonal).map(|x| (x / 2) as i32);
        self.offset = center - natural_center; self
    }

    pub fn build<S>(&self, structure: S) -> (S, SupercellToken)
    where
        S: Supercellable,
        S::Meta: Clone,
    {
        let (supercell, sc_token) = self.build_with(structure, |meta, _| meta.clone());
        (S::from_structure(supercell), sc_token)
    }

    pub fn build_with<S, M2, F>(&self, structure: S, make_meta: F) -> (Structure<M2>, SupercellToken)
    where
        S: Supercellable,
        F: FnMut(&S::Meta, [u32; 3]) -> M2,
    {
        let structure = structure.into_structure();
        diagonal_with(self.clone(), structure, make_meta)
    }
}

/// Compatability shim from Coords to old CoordStructure, which supercell
/// is still based around.
pub trait Supercellable {
    type Meta;

    fn into_structure(self) -> Structure<Self::Meta>;
    fn from_structure(structure: Structure<Self::Meta>) -> Self;
}

impl<M> Supercellable for Structure<M> {
    type Meta = M;

    fn into_structure(self) -> Structure<Self::Meta> { self }
    fn from_structure(structure: Structure<Self::Meta>) -> Self { structure }
}

impl Supercellable for Coords {
    type Meta = ();

    fn into_structure(self) -> Structure<Self::Meta> {
        self.with_uniform_metadata(())
    }
    fn from_structure(structure: Structure<Self::Meta>) -> Self {
        structure.coords
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
    let Structure { coords, meta } = structure;
    let Coords { lattice, coords } = coords;

    // construct a SupercellToken ASAP so that we know the rest of the code
    // works for general supercell matrices
    let sc = builder.into_sc_token(coords.len());

    let sc_carts = image_lattice_vecs(sc.periods, sc.offset, &lattice);
    let mut new_carts = Vec::with_capacity(sc.num_supercell_atoms());
    for atom_cart in coords.into_carts(&lattice) {
        let old_len = new_carts.len();
        new_carts.extend_from_slice(&sc_carts);
        ::util::translate_mut_n3_3(&mut new_carts[old_len..], &atom_cart);
    }

    let sc_idx = image_cells(sc.periods);
    let mut new_meta = Vec::with_capacity(sc.num_supercell_atoms());
    for m in meta {
        for &idx in &sc_idx {
            new_meta.push(make_meta(&m, idx));
        }
    }

    let structure = Structure {
        coords: Coords {
            lattice: &sc.integer_lattice * &lattice,
            coords: CoordsKind::Carts(new_carts),
        },
        meta: new_meta,
    };

    (structure, sc)
}

/// Contains enough information to deconstruct a supercell produced by this library.
///
/// **The order of the atoms in the supercell is unspecified** and may change with
/// the needs of rsp2. You must use the methods on this type if you need to convert
/// between index representations, or to work in the subgroup of cell images.
///
/// It provides a variety of methods for converting between various forms of indices,
/// summarized below along with the terms that frequently appear in method names:
///
/// * **`primitive_atom: usize`** -
///   index of a site in the original structure. `0 <= primitive_atom < num_primitive_atoms`.
///
/// * **`atom: usize`** -
///   index of a site in the superstructure. `0 <= atom < num_supercell_atoms`.
///
/// * **`cell: [u32; 3]`** -
///   the index of an image of the primitive structure. `0 <= cell[k] < periods[k]`.
///   Like site indices, these are considered to be unique identifiers, and are expected to
///   be in bounds. **The vector difference of two cells is a meaningless quantity.**
///
/// * **`lattice_point: V3<i32>`** -
///   a more mathematically-oriented form of `cell`. It is the integer coordinates of
///   the lattice vector that were added to a primitive site to produce a given supercell
///   site. In method outputs, this will be in the range
///   `0 <= lattice_point[k] - offset[k] < periods[k]`.  In method inputs, this will
///   generally be allowed to be any arbitrary integer vector.
pub struct SupercellToken {
    // Precomputed diagonal of the Hermite Normal Form of `integer_lattice`.
    periods: [u32; 3],
    // defines the range of `lattice_point`s
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

    #[inline]
    pub fn periods(&self) -> [u32; 3] {
        self.periods
    }

    /// The number of images taken of the unit cell contained in the supercell.
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.periods.iter().product::<u32>() as _
    }

    #[inline]
    pub fn num_primitive_atoms(&self) -> usize {
        self.num_primitive_atoms
    }

    #[inline]
    pub fn num_supercell_atoms(&self) -> usize {
        self.num_cells() * self.num_primitive_atoms()
    }

    // !!! This function affects the supercell convention !!! (SUPERCELL-CONV)
    // When modifying it, you must modify all functions that have this label.
    //
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
    pub fn deconstruct<S>(&self, validation_radius: f64, structure: S)
    -> Result<S, BigDisplacement>
    where
        S: Supercellable,
    {
        self.deconstruct_with(
            validation_radius,
            structure,
            |mut metas| metas.next().unwrap())
    }

    // !!! This function affects the supercell convention !!! (SUPERCELL-CONV)
    // When modifying it, you must modify all functions that have this label.
    //
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
    pub fn deconstruct_with<S, S2, F>(
        &self,
        validation_radius: f64,
        structure: S2,
        mut fold_meta: F,
    ) -> Result<S, BigDisplacement>
    where
        S: Supercellable,
        S2: Supercellable,
        F: FnMut(OwnedMetas<S2::Meta>) -> S::Meta,
    {
        let structure = structure.into_structure();
        assert_eq!(
            structure.num_atoms(), self.num_supercell_atoms(),
            "wrong # of atoms in supercell",
        );

        let num_cells = self.num_cells();
        let SupercellToken { periods, offset, ref integer_lattice, num_primitive_atoms } = *self;
        let Structure { coords, meta } = structure;
        let Coords { lattice, coords } = coords;

        let primitive_lattice = integer_lattice.inverse_matrix() * &lattice;

        let out_carts = {
            let neg_offsets = {
                let mut vs = image_lattice_vecs(periods, offset, &primitive_lattice);
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

        Ok(S::from_structure(Structure {
            coords: Coords {
                lattice: primitive_lattice,
                coords: CoordsKind::Carts(out_carts),
            },
            meta: out_meta,
        }))
    }

    fn check_cell(&self, cell: [u32; 3]) {
        for k in 0..3 {
            assert!(
                cell[k] < self.periods[k],
                "cell index {:?} out of bounds (periods: {:?})",
                cell, self.periods,
            )
        }
    }

    /// Get the cell index of the centermost cell.
    ///
    /// Ties on even-dimensioned axes are broken towards zero.
    ///
    /// If you need this, then you've probably written some dumb
    /// algorithm that only works with odd-dimensioned supercells
    /// that are at least 3x3x3 large. You should fix that.
    pub fn center_cell(&self) -> [u32; 3] {
        V3(self.periods).map(|x| x / 2).0
    }

    /// **Note:** The cell must be in bounds.
    pub fn lattice_point_from_cell(&self, cell: [u32; 3]) -> V3<i32> {
        self.check_cell(cell);
        self.lattice_point_from_cell_unchecked(cell)
    }

    pub fn lattice_point_from_cell_unchecked(&self, cell: [u32; 3]) -> V3<i32> {
        V3(cell).map(|x| x as i32) + self.offset
    }

    /// **Note:** The lattice point is wrapped into the supercell.
    pub fn cell_from_lattice_point(&self, v: V3<i32>) -> [u32; 3] {
        let diff = v - self.offset;
        let cell = V3::from_fn(|k| ::util::mod_euc(diff[k], self.periods[k] as i32) as u32);
        cell.0
    }

    /// **Note:** The cell must be in bounds.
    pub fn atom_from_cell(&self, prim: usize, cell: [u32; 3]) -> usize {
        assert!(prim < self.num_primitive_atoms);
        self.check_cell(cell);
        self.atom_from_cell_unchecked(prim, cell)
    }

    /// **Note:** The lattice point is wrapped into the supercell.
    pub fn atom_from_lattice_point(&self, prim: usize, lattice_point: V3<i32>) -> usize {
        let cell = self.cell_from_lattice_point(lattice_point);
        self.atom_from_cell_unchecked(prim, cell)
    }

    // !!! This function affects the supercell convention !!! (SUPERCELL-CONV)
    // When modifying it, you must modify all functions that have this label.
    pub fn atom_from_cell_unchecked(&self, prim: usize, cell: [u32; 3]) -> usize {
        use ::rsp2_array_types::dot;
        let [len_a, len_b, len_c] = self.periods;
        let stride_c = 1;
        let stride_b = stride_c * len_c as usize;
        let stride_a = stride_b * len_b as usize;
        let stride_prim = stride_a * len_a as usize;
        let strides = V3([stride_a, stride_b, stride_c]);

        prim * stride_prim + dot(&strides, &V3(cell).map(|x| x as usize))
    }

    /// Defines which image of the primitive cell each atom in the supercell belongs to.
    ///
    /// Each image of the primitive cell is represented by the integral coordinates of the
    /// lattice vector used to produce it. It is guaranteed that the output will contain
    /// `self.num_cells()` unique values, and that each one will appear exactly
    /// `self.num_primitive_atoms()` times.
    pub fn atom_cells(&self) -> Vec<[u32; 3]> {
        image_cells(self.periods).iter().cloned()
            .cycle().take(self.num_supercell_atoms())
            .collect()
    }

    /// Gives coefficients of the lattice vectors that were added to each atom in the supercell.
    ///
    /// This is like `cell_indices`, but gives the indices as signed integer offsets.
    /// The `offset` from building the supercell affects this, so that e.g. centered supercells
    /// have a signed index of `[0, 0, 0]` assigned to atoms in the centermost unit cell.
    pub fn atom_lattice_points(&self) -> Vec<V3<i32>> {
        image_lattice_points(self.periods, self.offset).iter().cloned()
            .cycle().take(self.num_supercell_atoms())
            .collect()
    }

    /// Defines the primitive site corresponding to each supercell site.
    pub fn atom_primitive_atoms(&self) -> Vec<usize> {
        self.replicate(&(0..self.num_primitive_atoms).collect::<Vec<_>>())
    }

    // !!! This function affects the supercell convention !!! (SUPERCELL-CONV)
    // When modifying it, you must modify all functions that have this label.
    //
    /// Get a depermutation representing translation by a primitive cell lattice point.
    ///
    /// Please see `conventions.md` for an explanation of depermutations.
    pub fn lattice_point_translation_deperm(&self, lattice_point: V3<i32>) -> Perm {
        // Depermutations that permute the cells along each axis independently.
        // (expressed in the quotient spaces of images along those axes)
        let axis_deperms: [_; 3] = arr_from_fn(|k| {
            Perm::eye(self.periods[k] as usize)
                .shift_signed(lattice_point[k] as isize)
        });

        // Construct the overall deperm as an outer product of deperms.
        // this could be written as a fold, but I wanted to emphasize the order;
        // the innermost perm must correspond to the fastest-varying index.
        Perm::eye(self.num_primitive_atoms)
            .with_inner(&axis_deperms[0])
            .with_inner(&axis_deperms[1])
            .with_inner(&axis_deperms[2])
    }
}

//--------------------------------
// functions prefixed with 'image' describe the quotient space of unit cell images.
// (which is of size 'periods.iter().product()')

// !!! This function affects the supercell convention !!! (SUPERCELL-CONV)
// When modifying it, you must modify all functions that have this label.
fn image_cells(periods: [u32; 3]) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(periods.iter().product::<u32>() as usize);
    let [na, nb, nc] = periods;
    for ia in 0..na {
        for ib in 0..nb {
            for ic in 0..nc {
                out.push([ia, ib, ic]);
            }
        }
    }
    out
}

fn image_lattice_points(periods: [u32; 3], offset: V3<i32>) -> Vec<V3<i32>> {
    image_cells(periods).into_iter()
        .map(|idx| V3(idx).map(|x| x as i32) + offset)
        .collect()
}

fn image_lattice_vecs(periods: [u32; 3], offset: V3<i32>, lattice: &Lattice) -> Vec<V3> {
    image_lattice_points(periods, offset).into_iter()
        .map(|idx| idx.map(|x| x as f64) * lattice)
        .collect()
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use ::rsp2_soa_ops::{Permute, Perm};
    use ::{Coords, CoordsKind, Structure, Lattice};
    use ::rsp2_array_types::{V3, Envee};

    use ::rand::Rng;

    #[test]
    fn diagonal_supercell_smoke_test() {
        let coords = CoordsKind::Fracs(vec![[0.0, 0.0, 0.0]].envee());

        let original = Coords::new(Lattice::eye(), coords);
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

        let original = Coords::new(lattice, coords);
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

        let original = Coords::new(lattice, coords);
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
    fn cell_index_conversions() {
        let sc_token = ::supercell::diagonal([2, 5, 3]).into_sc_token(7);
        let lattice_points = sc_token.atom_lattice_points();
        let cells = sc_token.atom_cells();
        ::itertools::assert_equal(
            lattice_points.iter().cloned(),
            cells.iter().map(|&v| sc_token.lattice_point_from_cell(v)),
        );
        ::itertools::assert_equal(
            cells.iter().cloned(),
            lattice_points.iter().map(|&v| sc_token.cell_from_lattice_point(v)),
        );
        // lattice points are automatically wrapped
        let offset = V3([-6, 35, 9]); // a supercell lattice point
        ::itertools::assert_equal(
            cells.iter().cloned(),
            lattice_points.iter().map(|&v| sc_token.cell_from_lattice_point(v + offset)),
        );
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
                let sorting_perm = Perm::argsort(structure.metadata());
                structure = structure.permuted_by(&sorting_perm);
                structure.reduce_positions();
                structure.to_carts()
            };
            assert_eq!(
                canonicalized_coords(translated),
                canonicalized_coords(depermuted),
            );
        }
    }


    #[test]
    fn conversion_methods_are_consistent() {
        let mut rng = ::rand::thread_rng();
        for trial in 0..10 {
            let num_prim = rng.gen_range(1, 9 + 1);
            let lattice = Lattice::eye();
            let coords = Coords::new(lattice, CoordsKind::Carts(vec![V3::zero(); num_prim]));
            let dims = match trial {
                0|2 => [1, 1, 1],
                _ => V3::from_fn(|_| rng.gen_range(1, 6 + 1)).0,
            };
            let center = match trial {
                0|1 => V3::zero(),
                _ => V3::from_fn(|_| rng.gen_range(-10, 10 + 1)),
            };
            let (_, sc_token) = ::supercell::diagonal(dims).center(center).build(coords);

            let lattice_points = sc_token.atom_lattice_points();
            let cells = sc_token.atom_cells();
            let prims = sc_token.atom_primitive_atoms();

            // a lattice point in the supercell lattice, to test wrapping
            let test_offset = {
                let signs = V3::from_fn(|_| rng.gen_range(0, 2) - 1);
                let multiples = V3::from_fn(|_| rng.gen_range(1, 5));
                V3::from_fn(|k| sc_token.periods[k] as i32 * multiples[k] * signs[k])
            };

            assert_eq!(sc_token.num_supercell_atoms(), lattice_points.len());
            for atom in 0..lattice_points.len() {
                assert_eq!(
                    atom,
                    sc_token.atom_from_cell(prims[atom], cells[atom]),
                );
                assert_eq!(
                    atom,
                    sc_token.atom_from_lattice_point(prims[atom], lattice_points[atom]),
                );
                assert_eq!(
                    atom,
                    sc_token.atom_from_lattice_point(prims[atom], lattice_points[atom] + test_offset),
                );
            }
        }
    }
}
