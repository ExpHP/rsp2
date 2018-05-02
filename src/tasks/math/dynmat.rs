#![allow(unused)] // FIXME

use ::FailResult;
use ::rsp2_array_types::{V3, M33, M3};
use ::rsp2_structure::{Perm};
use ::rsp2_structure::supercell::SupercellToken;
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
    // `deperm` should describe the symmop as a depermutation
    //
    // see conventions.md for info about depermutations.
    fn derive_from_symmetry(
        &self,
        sc_token: &SupercellToken,
        cart_rot: &M33,
        deperm: &Perm,
    ) -> Self {

        let (atom_displaced, atom_affected) = {
            let cells = sc_token.signed_cell_indices();
            ::util::zip_eq(&self.atom_displaced, &self.atom_displaced)
                .map(|(&displaced, &affected)| {
                    // rotate the atoms to a new index
                    let mut displaced = deperm[displaced];
                    let mut affected = deperm[affected];

                    // make sure 'displaced' is in the center cell
                    let cell = cells[displaced as usize];
                    if cell != V3([0, 0, 0]) {
                        // FIXME I imagine this has the potential to be very slow,
                        //       maybe we should make an intermediate data structure
                        //       with all the lattice point deperms
                        let inv_deperm = sc_token.lattice_point_translation_deperm(-cell);
                        affected = inv_deperm[affected];
                        displaced = inv_deperm[displaced];
                    }
                    assert_eq!(cells[displaced as usize], V3([0, 0, 0]));

                    (displaced, affected)
                }).unzip()
        };

        // rotate the cartesian vectors
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

    fn solve_force_constants(&self, sc_token: &SupercellToken) -> FailResult<ForceConstants>
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
            use rsp2_linalg::{CMatrix, left_pseudoinverse, dot};
            //
            //    F = -U Phi
            //
            // * Phi is the 3x3 matrix of force constants for this pair of atoms
            // * F is the Nx3 matrix of forces experienced by 'affected'
            // * U is the Nx3 matrix of corresponding displacements for 'displaced'
            //
            // for large enough N (and assuming sufficient rank),
            // we can solve for Phi using the pseudoinverse
            assert!(forces.len() > 3, "not enough FCs? (got {})", forces.len());
            let displacements: CMatrix = displacements.into();
            let forces: CMatrix = forces.into();
            let phi: CMatrix = dot(&*left_pseudoinverse(displacements)?, &*forces).into();
            row_atom.push(displaced as usize);
            col_atom.push(affected as usize);
            cart_matrix.push(M3(phi.c_order_data().nest().to_array()));
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


