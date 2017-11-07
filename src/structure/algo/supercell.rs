use ::{Structure, Lattice, Coords};
use ::{Result, Error, ErrorKind};

use ::ordered_float::NotNaN;
use ::rsp2_array_utils::{dot, try_arr_from_fn};

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
    let Structure { lattice, coords, meta } = structure;

    let integer_lattice = Lattice::orthorhombic(dims.0 as f64, dims.1 as f64, dims.2 as f64);

    // number of offsets along each lattice vector.
    // trivial for a diagonal supercell; less so for a general supercell
    let periods = [dims.0, dims.1, dims.2];
    let num_sc = (periods[0] * periods[1] * periods[2]) as usize;
    let final_size = num_sc * coords.len();

    let sc_carts = sc_lattice_vecs(periods, &lattice);
    let mut new_carts = Vec::with_capacity(final_size);
    for atom_cart in coords.into_carts(&lattice) {
        let old_len = new_carts.len();
        new_carts.extend_from_slice(&sc_carts);
        ::util::translate_mut_n3_3(&mut new_carts[old_len..], &atom_cart);
    }

    let sc_idx = sc_indices(periods);
    let mut new_meta = Vec::with_capacity(final_size);
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

    let token = SupercellToken { periods, integer_lattice };
    (structure, token)
}

/// Contains enough information to deconstruct a supercell produced by this library.
pub struct SupercellToken {
    periods: [u32; 3],
    // supercell in units of primitive cell vectors.  Elements are integral
    integer_lattice: Lattice,
}

pub type OwnedMetas<'a,T> = ::std::vec::Drain<'a,T>;
impl SupercellToken {

    pub fn num_cells(&self) -> usize {
        (self.periods[0] * self.periods[1] * self.periods[2]) as usize
    }

    /// Takes data for each atom of the primitive cell and expands it to the
    /// size of the supercell.
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
        let SupercellToken { periods, ref integer_lattice } = *self;
        let Structure { lattice, coords, meta } = structure;
        let num_sc = (periods[0] * periods[1] * periods[2]) as usize;
        ensure!(coords.len() % num_sc == 0, "wrong # of atoms in supercell");
        let num_atoms = coords.len() / num_sc;

        let primitive_lattice = integer_lattice.inverse_matrix() * &lattice;

        let out_carts = {
            let neg_offsets = {
                use ::slice_of_array::prelude::*;
                let mut v = sc_lattice_vecs(periods, &primitive_lattice);
                for x in v.flat_mut() {
                    *x *= -1.0;
                }
                v
            };

            let mut carts = coords.into_carts(&lattice);
            let mut image_carts = Vec::with_capacity(num_sc);
            let mut out_carts = Vec::with_capacity(num_atoms);
            while !carts.is_empty() {
                // Fold all images of a single atom
                let new_len = carts.len() - num_sc;

                image_carts.clear();
                image_carts.extend(carts.drain(new_len..));

                ::util::translate_mut_n3_n3(&mut image_carts, &neg_offsets);

                out_carts.push(try_arr_from_fn(|k| {
                    // sigh
                    let not_nans = image_carts.iter().map(|v| NotNaN::new(v[k]).unwrap()).collect::<Vec<_>>();

                    let min = not_nans.iter().cloned().min().unwrap().into_inner();
                    let max = not_nans.iter().cloned().max().unwrap().into_inner();
                    ensure!(
                        max - min <= 2.0 * validation_radius,
                        ErrorKind::BigDisplacement(max - min));

                    let sum = not_nans.into_iter().map(NotNaN::into_inner).sum::<f64>();
                    Ok::<_, Error>(sum / num_sc as f64)
                })?);
            }
            // Atoms were done in reverse order
            out_carts.into_iter().rev().collect()
        };

        let out_meta = {
            let mut meta = meta;
            let mut out_meta = Vec::with_capacity(num_atoms);
            while !meta.is_empty() {
                // Fold all images of a single atom
                let new_len = meta.len() - num_sc;
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
}

// supercell indices in the library's preferred order
fn sc_indices(periods: [u32; 3]) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity((periods[0] * periods[1] * periods[2]) as usize);
    for ia in 0..periods[0] {
        for ib in 0..periods[1] {
            for ic in 0..periods[2] {
                out.push([ia, ib, ic]);
            }
        }
    }
    out
}

// supercell image offsets in the library's preferred order
fn sc_lattice_vecs(periods: [u32; 3], lattice: &Lattice) -> Vec<[f64; 3]> {
    sc_indices(periods).into_iter()
        .map(|idx| dot(&[idx[0] as f64, idx[1] as f64, idx[2] as f64], lattice.matrix()))
        .collect()
}

#[cfg(test)]
#[deny(unused)]
mod tests {

    #[test]
    fn diagonal_supercell_smoke_test() {
        use ::{Coords, Structure, Lattice};

        let coords = Coords::Fracs(vec![[0.0, 0.0, 0.0]]);

        let original = Structure::new_coords(Lattice::eye(), coords);
        let (supercell, sc_token) = ::supercell::diagonal((2, 2, 2), original.clone());

        assert_eq!(supercell.num_atoms(), 8);
        assert_eq!(supercell.lattice(), &Lattice::cubic(2.0));

        assert!(::util::eq_unordered_n3(&supercell.to_carts(), &[
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
        ]));

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

        let coords = Coords::Fracs(vec![
            [ 0.5, -0.5, 0.0], // cart: [+1.0, -1.0,  0.0]
            [ 0.0,  0.5, 0.5], // cart: [ 0.0, +1.0, +1.0]
        ]);

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
