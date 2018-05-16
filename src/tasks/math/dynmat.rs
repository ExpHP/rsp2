#![allow(unused)] // FIXME

use ::FailResult;
use ::rsp2_array_types::{V3, M33, M3};
use ::rsp2_soa_ops::{Perm, Permute};
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
    atom_displaced: Vec<usize>,
    atom_affected: Vec<usize>,
    cart_force: Vec<V3>,
    cart_displacement: Vec<V3>,
}

// Displaced atoms are always in this cell.
// (HACK: shouldn't need to be pub(crate))
pub(crate) const DISPLACED_CELL: [u32; 3] = [0, 0, 0];

impl ForceSets {
    pub fn from_displacement(
        // (atom_displaced, cart_displacement)
        displacement: (usize, V3),
        // Item = (atom_affected, cart_force)
        forces: impl IntoIterator<Item=(usize, V3)>,
    ) -> Self {
        let (atom_displaced, cart_displacement) = displacement;
        let (atom_affected, cart_force): (Vec<_>, _) = forces.into_iter().unzip();

        let count = atom_affected.len();
        let atom_displaced = vec![atom_displaced; count];
        let cart_displacement = vec![cart_displacement; count];
        ForceSets { atom_affected, atom_displaced, cart_force, cart_displacement }
    }

    // Picture the following:
    //
    // * Suppose that our structure were printed out, so that we may lay a transparency
    //   sheet over it. (also suppose that the structure were 2D, so that the printout could
    //   faithfully render it without perspective)
    // * Draw arrows on all atoms representing the present forces, and draw a circle around
    //   the displaced atom.  This is the input set of data.
    // * Now apply `cart_rot` to the transparency sheet.
    //   **This is the output set of data.**
    //
    // `super_deperm` should describe the symmop as a depermutation of the supercell.
    //
    // see conventions.md for info about depermutations.
    //
    // FIXME shouldn't be pub(crate)
    // FIXME there is a better strategy which lets us do only one pseudoinversion per unique
    //       displacement vector.  Basically, once all space group operators are accounted
    //       for, we will have one force per space group op in every interacting pair,
    //       so we simply need to make sure they are always in the same order.
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
                    let displaced = super_deperm.permute_index(displaced);
                    let affected = super_deperm.permute_index(affected);

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
    pub(crate) fn solve_force_constants(&self, sc_token: &SupercellToken
                                        , perm_HACK: &Perm
        , carts_HACK: &[V3]
    ) -> FailResult<ForceConstants>
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
//
//            if forces.flat().iter().any(|&x| f64::abs(x) > 1e-7) && displaced == 0 {
//
//                let V3([x, y, z]) = carts_HACK[affected];
//                info!("displacements {} {} [{}, {}, {}]", perm_HACK[displaced], perm_HACK[affected], x, y, z);
//                use ::rsp2_array_types::Unvee;
//    //            let mut show = (&displacements[..]).unvee().to_vec();
//                //show.sort_by(|a, b| a.partial_cmp(b).unwrap());
//                for (V3([ux, uy, uz]), V3([fx, fy, fz])) in ::util::zip_eq(&displacements, &forces) {
//                    info!(" [{:+.06}, {:+.06}, {:+.06}, {:+.06}, {:+.06}, {:+.06}]", ux, uy, uz, fx, fy, fz);
//                }
//            }

            let displacements: CMatrix = displacements.into();
            let forces: CMatrix = forces.into();
            let phi: CMatrix = dot(&*left_pseudoinverse(displacements)?, &*forces).into();
            row_atom.push(displaced as usize);
            col_atom.push(affected as usize);
            cart_matrix.push(-&M3(phi.c_order_data().nest::<V3>().to_array()));
        }

        Ok(ForceConstants { row_atom, col_atom, cart_matrix })
    }
}

#[derive(Debug, Clone)]
pub struct ForceConstants {
    row_atom: Vec<usize>,
    col_atom: Vec<usize>,
    // this might be awkward to work with...
    cart_matrix: Vec<M33>,
}

impl ForceConstants {
    pub(crate) fn symmetrize_by_transpose(mut self) -> Self {
        let t = self.clone().transpose();
        self.extend(t);
        for m in &mut self.cart_matrix {
            *m *= 0.5;
        }
        self
    }

    /// Transpose as if it were a 3N by 3N matrix.
    pub(crate) fn transpose(self) -> Self {
        let ForceConstants { row_atom, col_atom, cart_matrix } = self;
        let cart_matrix = cart_matrix.into_iter().map(|m| m.t()).collect();
        ForceConstants {
            row_atom: col_atom,
            col_atom: row_atom,
            cart_matrix,
        }
    }

    pub(crate) fn extend(&mut self, other: Self) {
        self.row_atom.extend(other.row_atom);
        self.col_atom.extend(other.col_atom);
        self.cart_matrix.extend(other.cart_matrix);
    }

    /// Ensure that at most a single item per atom pair exists by summing duplicates.
    pub(crate) fn canonicalize(self) -> Self {
        let ForceConstants { row_atom, col_atom, cart_matrix } = self;
        // sort row major
        let perm = {
            let sort_data = zip_eq!(&row_atom, &col_atom).collect::<Vec<_>>();
            Perm::argsort(&sort_data)
        };
        let row_atom = row_atom.permuted_by(&perm);
        let col_atom = col_atom.permuted_by(&perm);
        let cart_matrix = cart_matrix.permuted_by(&perm);

        // reduce
        let iter = zip_eq!(zip_eq!(row_atom, col_atom), cart_matrix);
        let iter = IntoReducedIter::new(iter, ::std::ops::Add::add);
        let (pos, cart_matrix): (Vec<_>, _) = iter.unzip();
        let (row_atom, col_atom) = pos.into_iter().unzip();

        ForceConstants { row_atom, col_atom, cart_matrix }
    }
}

use self::reduce_items::IntoReducedIter;
mod reduce_items {
    // from an old project

    /// Iterator adapter that sums consecutive consecutive Ts with the same key pair.
    #[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
    pub struct IntoReducedIter<K, T, F> {
        iter: ::std::iter::Peekable<::std::vec::IntoIter<(K, T)>>,
        function: F,
    }

    impl<K: Eq, T, F> IntoReducedIter<K, T, F>
    where
        F: FnMut(T, T) -> T,
    {
        #[inline]
        pub fn new(iter: impl IntoIterator<Item=(K, T)>, function: F) -> Self {
            IntoReducedIter {
                iter: iter.into_iter().collect::<Vec<_>>().into_iter().peekable(),
                function,
            }
        }
    }

    impl<K: Eq, T, F> Iterator for IntoReducedIter<K, T, F>
    where
        K: Copy, // greatly simplifies a borrowck issue
        F: FnMut(T, T) -> T,
    {
        type Item = (K, T);
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().and_then(|(pos, mut val)| {
                while let Some(&(next_pos, _)) = self.iter.peek() {
                    if next_pos == pos {
                        val = (self.function)(val, self.iter.next().unwrap().1);
                    } else {
                        break;
                    }
                }
                Some((pos, val))
            })
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            // any number of elements may be reduced together
            let (lo, hi) = self.iter.size_hint();
            (::std::cmp::min(lo, 1), hi)
        }
    }
}


impl ForceConstants {
    /// For debugging purposes only...
    #[allow(unused)]
    pub(crate) fn to_dense_matrix(&self, num_supercell_atoms: usize) -> Vec<Vec<[[f64; 3]; 3]>> {
        use ::rsp2_array_types::Unvee;
        let n_basis = num_supercell_atoms;
        let mut out = vec![vec![[[0.0f64; 3]; 3]; n_basis]; n_basis];
        for ((&row_atom, &col_atom), matrix) in ::util::zip_eq(::util::zip_eq(&self.row_atom, &self.col_atom), &self.cart_matrix) {
            //out[row_atom][col_atom] = matrix.unvee();
            for row in 0..3 {
                for col in 0..3 {
                    out[row_atom][col_atom][row][col] += matrix[row][col];
                }
            }
        }
        out
    }
}

