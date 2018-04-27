use ::{Result, ErrorKind};
use ::rsp2_array_types::{V3, M33, M3, dot};
use ::rsp2_structure::{FracOp, Perm};
use ::std::collections::HashMap;
use ::slice_of_array::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct ForceSets {
    atom_displaced: Vec<u32>,
    atom_affected: Vec<u32>,
    cart_force: Vec<V3>,
    cart_displacement: Vec<V3>,
}

impl ForceSets {
    // `perm` should describe the symmop as a permutation, such that applying the
    // operator moves the atom at `coords[i]` to `coords[perm[i]]`
    fn derive_from_symmetry(&self, cart_rot: &M33, perm: &Perm) -> Self {
        let atom_displaced = self.atom_displaced.iter().map(|&i| perm[i]).collect();
        let atom_affected = self.atom_affected.iter().map(|&i| perm[i]).collect();
        let cart_op_t = cart_rot.t();
        let cart_force = self.cart_force.iter().map(|v| v * &cart_op_t).collect();
        let cart_displacement = self.cart_displacement.iter().map(|v| v * &cart_op_t).collect();
        ForceSets { atom_displaced, atom_affected, cart_force, cart_displacement }
    }

    fn concat_from<Ss>(iter: Ss) -> Self
    where Ss: IntoIterator<Item=ForceSets>,
    {
        iter.into_iter().fold(Self::default(), |mut a, b| {
            a.atom_affected.extend(b.atom_affected);
            a.atom_displaced.extend(b.atom_displaced);
            a.cart_force.extend(b.cart_force);
            a.cart_displacement.extend(b.cart_displacement);
            a
        })
    }

    /// NOTE: The error case is for singular matrices.
    /// (I wish I was using 'failure' right now...)
    fn solve_force_constants(&self) -> Result<ForceConstants>
    {
        use ::util::zip_eq as z;
        let ForceSets {
            ref atom_displaced,
            ref atom_affected,
            ref cart_force,
            ref cart_displacement,
        } = *self;

        let mut map = HashMap::new();

        // build a (likely overconstrained) system of equations for each interacting (i,j) pair
        z(z(z(atom_displaced, atom_affected), cart_force), cart_displacement)
            .for_each(|(((&displaced, &affected), force), displacement)| {
                let key = (displaced, affected);
                let entry = map.entry(key).or_insert((vec![], vec![]));
                let &mut (ref mut fs, ref mut us) = entry;
                fs.push(*force);
                us.push(*displacement);
            });

        let mut row_atom = vec![];
        let mut col_atom = vec![];
        let mut cart_matrix = vec![];
        for ((displaced, affected), (forces, displacements)) in map {
            //
            //    F = -U Phi
            //
            // * Phi is the 3x3 matrix of force constants for this pair of atoms
            // * F is the Nx3 matrix of forces experienced by 'affected'
            // * U is the Nx3 matrix of corresponding displacements for 'displaced'
            //
            // for large enough N (and assuming sufficient rank),
            // we can solve for Phi using the pseudoinverse
            assert!(forces.len() > 6, "not enough FCs? (got {})", forces.len());
            let displacements: Matrix = (&displacements[..]).into();
            let forces: Matrix = (&forces[..]).into();
            let phi = &linalg::left_pseudoinverse(&displacements)? * &forces;
            row_atom.push(displaced as usize);
            col_atom.push(affected as usize);
            cart_matrix.push(M3(*phi.row_major_data().nest().as_array()));
        }

        Ok(ForceConstants { row_atom, col_atom, cart_matrix })
    }
}

pub struct ForceConstants {
    row_atom: Vec<usize>,
    col_atom: Vec<usize>,
    // this might be awkward to work with...
    cart_matrix: Vec<M33>,
}


