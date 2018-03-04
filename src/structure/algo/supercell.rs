#![allow(deprecated)] // HACK: reduce warning-spam at call sites, I only care about the `use`

use ::{Structure, Lattice, Coords};
use ::{Result, Error, ErrorKind};

#[warn(deprecated)]
use ::rsp2_array_utils::{dot, try_arr_from_fn};

use ::rsp2_array_types::V3;

pub fn diagonal<M>(dims: (u32,u32,u32), structure: Structure<M>)
-> (Structure<M>, SupercellToken)
where M: Clone,
{
    diagonal_with(dims, structure, |meta,_| meta.clone())
}

pub fn diagonal_with<M, F>(dims: (u32,u32,u32), structure: Structure<M>, mut make_meta: F)
-> (Structure<M>, SupercellToken)
where F: FnMut(&M, (u32,u32,u32)) -> M,
{
    let num_primitive_atoms = structure.num_atoms();
    let Structure { lattice, coords, meta } = structure;

    let integer_lattice = Lattice::orthorhombic(dims.0 as f64, dims.1 as f64, dims.2 as f64);

    // number of offsets along each lattice vector.
    // trivial for a diagonal supercell; less so for a general supercell
    let periods = [dims.0, dims.1, dims.2];
    let num_sc = (periods[0] * periods[1] * periods[2]) as usize;
    let num_supercell_atoms = num_sc * coords.len();

    let sc_carts = sc_lattice_vecs(periods, &lattice);
    let mut new_carts = Vec::with_capacity(num_supercell_atoms);
    for atom_cart in coords.into_carts(&lattice) {
        let old_len = new_carts.len();
        new_carts.extend_from_slice(&sc_carts);
        ::util::translate_mut_n3_3(&mut new_carts[old_len..], &atom_cart);
    }

    let sc_idx = sc_indices(periods);
    let mut new_meta = Vec::with_capacity(num_supercell_atoms);
    for m in meta {
        for idx in &sc_idx {
            new_meta.push(make_meta(&m, (idx[0], idx[1], idx[2])));
        }
    }

    let structure = Structure {
        lattice: &integer_lattice * &lattice,
        coords: Coords::Carts(new_carts),
        meta: new_meta,
    };

    let token = SupercellToken { periods, integer_lattice, num_primitive_atoms };
    (structure, token)
}

/// Contains enough information to deconstruct a supercell produced by this library.
///
/// The order of the atoms in the supercell is unspecified and may change; you should
/// not assume that the images are grouped by unit cell or by primitive site.
/// Instead, please refer to the methods `cell_indices` and `primitive_site_indices`.
pub struct SupercellToken {
    periods: [u32; 3],
    num_primitive_atoms: usize,
    // supercell in units of primitive cell vectors.  Elements are integral
    integer_lattice: Lattice,
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
    -> Result<Structure<M>>
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
    pub fn deconstruct_with<M, F>(&self, validation_radius: f64, structure: Structure<M>, mut fold_meta: F)
    -> Result<Structure<M>>
    where F: FnMut(OwnedMetas<M>) -> M,
    {
        ensure!(structure.num_atoms() == self.num_supercell_atoms(),
            "wrong # of atoms in supercell");

        let num_cells = self.num_cells();
        let SupercellToken { periods, ref integer_lattice, num_primitive_atoms } = *self;
        let Structure { lattice, coords, meta } = structure;

        let primitive_lattice = integer_lattice.inverse_matrix() * &lattice;

        let out_carts = {
            let neg_offsets = {
                let mut vs = sc_lattice_vecs(periods, &primitive_lattice);
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
                    ensure!(
                        max - min <= 2.0 * validation_radius,
                        ErrorKind::BigDisplacement(max - min));

                    let sum = this_axis().sum::<f64>();
                    Ok::<_, Error>(sum / num_cells as f64)
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
            coords: Coords::Carts(out_carts),
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

    /// Defines which atom in the primitive cell that each supercell atom is an image of.
    ///
    /// This function is constant for any given supercell; it merely exists to define the
    /// conventions for ordering used by the library.
    pub fn primitive_site_indices(&self) -> Vec<usize> {
        self.replicate(&(0..self.num_primitive_atoms).collect::<Vec<_>>())
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

// supercell image offsets in the library's preferred order
fn sc_lattice_vecs(periods: [u32; 3], lattice: &Lattice) -> Vec<V3> {
    sc_indices(periods).into_iter()
        .map(|idx| V3(dot(&idx.map(|x| x as f64).0, lattice.matrix())))
        .collect()
}

#[cfg(test)]
#[deny(unused)]
mod tests {

    use ::rsp2_array_types::envee;

    #[test]
    fn diagonal_supercell_smoke_test() {
        use ::{Coords, Structure, Lattice};

        let coords = Coords::Fracs(envee(vec![[0.0, 0.0, 0.0]]));

        let original = Structure::new_coords(Lattice::eye(), coords);
        let (supercell, sc_token) = ::supercell::diagonal((2, 2, 2), original.clone());

        assert_eq!(supercell.num_atoms(), 8);
        assert_eq!(supercell.lattice(), &Lattice::cubic(2.0));

        assert!(::util::eq_unordered_n3(&supercell.to_carts(), envee(&[
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
        ])));

        let deconstructed = sc_token.deconstruct(1e-10, supercell).unwrap();

        assert_eq!(original.to_carts(), deconstructed.to_carts());
        assert_eq!(original.lattice(), deconstructed.lattice());
    }

    #[test]
    fn test_diagonal_supercell() {
        use ::{Coords, Structure, Lattice};

        // nondiagonal lattice so that matrix multiplication order matters
        let lattice = [
            [2.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ];

        let coords = Coords::Fracs(envee(vec![
            [ 0.5, -0.5, 0.0], // cart: [+1.0, -1.0,  0.0]
            [ 0.0,  0.5, 0.5], // cart: [ 0.0, +1.0, +1.0]
        ]));

        let original = Structure::new_coords(Lattice::new(&lattice), coords);
        let (supercell, sc_token) = ::supercell::diagonal((4, 2, 2), original.clone());
        let deconstructed = sc_token.deconstruct(1e-10, supercell.clone()).unwrap();

        assert_eq!(original.to_carts(), deconstructed.to_carts());
        assert_eq!(original.lattice(), deconstructed.lattice());

        // test error on unequal shifts
        let mut supercell = supercell;
        let mut carts = supercell.to_carts();
        carts[4][1] += 1e-6;
        supercell.coords = Coords::Carts(carts);
        assert!(sc_token.deconstruct(1e-10, supercell.clone()).is_err());
    }
}
