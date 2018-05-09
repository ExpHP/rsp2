#![allow(unused)] // FIXME

use ::FailResult;
use ::rsp2_array_types::{V3, M33, M3};
use ::rsp2_structure::{Perm};
use ::rsp2_structure::supercell::SupercellToken;
use ::std::collections::HashMap;
use ::slice_of_array::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct ForceSets {
    // These are both indices in the supercell.
    //
    // Invariant: the displaced atom always has an index where `cell = DISPLACED_CELL`.
    //
    // (technically we *could* store primitive sites in `atom_displaced`...
    //  but I feel that storing supercell sites is safer, because it forces us
    //  to be wary of the fact that rotations might move `atom_displaced` to a
    //  different cell. (which is a fact we must account for in `atom_affected`))
    //
    // FIXME shouldn't need to be pub(crate)
    pub(crate) atom_displaced: Vec<usize>,
    pub(crate) atom_affected: Vec<usize>,
    pub(crate) cart_force: Vec<V3>,
    pub(crate) cart_displacement: Vec<V3>,
}

// Displaced atoms are always in this cell.
// (HACK: shouldn't need to be pub(crate))
pub(crate) const DISPLACED_CELL: [u32; 3] = [0, 0, 0];

impl ForceSets {
    // `super_deperm` should describe the symmop as a depermutation of the supercell.
    //
    // see conventions.md for info about depermutations.
    // FIXME shouldn't be pub(crate)
    pub(crate) fn derive_from_symmetry(
        &self,
        sc: &SupercellToken,
        cart_rot: &M33,
        super_deperm: &Perm,
    ) -> Self {
        assert_eq!(super_deperm.len(), sc.num_supercell_atoms());

        let primitive_atoms = sc.atom_primitive_atoms();
        let lattice_points = sc.atom_lattice_points();
        let expected_lattice_point = sc.lattice_point_from_cell(DISPLACED_CELL);

        let (atom_displaced, atom_affected) = {
            ::util::zip_eq(&self.atom_displaced, &self.atom_affected)
                .map(|(&displaced, &affected)| {
                    assert_eq!(expected_lattice_point, lattice_points[displaced as usize]);

                    // Rotate the atoms to a new index.
                    let displaced = super_deperm[displaced];
                    let affected = super_deperm[affected];

                    // 'displaced' might have been moved to a new cell.
                    // Translate the atoms to move it back.
                    let correction = expected_lattice_point - lattice_points[displaced as usize];
                    let translate_atom = |old_super| {
                        let old_super = old_super as usize;
                        sc.atom_from_lattice_point(
                            primitive_atoms[old_super],
                            lattice_points[old_super] + correction,
                        )
                    };
                    let displaced = translate_atom(displaced);
                    let affected = translate_atom(affected);

                    assert_eq!(expected_lattice_point, lattice_points[displaced as usize]);
                    (displaced, affected)
                }).unzip()
        };

        // rotate the cartesian vectors
        let cart_op_t = cart_rot.t();
        let cart_force = self.cart_force.iter().map(|v| v * &cart_op_t).collect();
        let cart_displacement = self.cart_displacement.iter().map(|v| v * &cart_op_t).collect();

        ForceSets { atom_displaced, atom_affected, cart_force, cart_displacement }
    }

    // FIXME shouldn't be pub(crate)
    pub(crate) fn concat_from<Ss>(iter: Ss) -> Self
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

    // FIXME shouldn't be pub(crate)
    pub(crate) fn solve_force_constants(&self, sc_token: &SupercellToken) -> FailResult<ForceConstants>
    {
        use ::util::zip_eq as z;
        let ForceSets {
            atom_displaced,
            atom_affected,
            cart_force,
            cart_displacement,
        } = self;

        let mut map = HashMap::new();

        // build a (likely overconstrained) system of equations for each interacting (i,j) pair
        z(z(z(atom_displaced, atom_affected), cart_force), cart_displacement)
            .for_each(|(((&displaced, &affected), force), displacement)| {
                let key = (displaced, affected);
                let entry = map.entry(key).or_insert((vec![], vec![]));
                let (fs, us) = entry;
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


impl ForceConstants {
    /// For debugging purposes only...
    #[allow(unused)]
    pub(crate) fn to_dense_matrix(&self, num_supercell_atoms: usize) -> Vec<Vec<f64>> {
        let n_basis = num_supercell_atoms * 3;
        let mut out = vec![vec![0.0f64; n_basis]; n_basis];
        for ((&row_atom, &col_atom), matrix) in ::util::zip_eq(::util::zip_eq(&self.row_atom, &self.col_atom), &self.cart_matrix) {
            let col_start = col_atom * 3;
            let col_end   = (col_atom + 1) * 3;
            for row_k in 0..3 {
                out[row_atom * 3 + row_k][col_start..col_end].copy_from_slice(&matrix[row_k].0);
            }
        }
        out
    }
}

