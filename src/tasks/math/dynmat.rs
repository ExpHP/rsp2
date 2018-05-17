#![allow(unused)] // FIXME

use ::FailResult;
use ::rsp2_array_types::{V3, M33, M3, mat};
use ::rsp2_soa_ops::{Perm, Permute};
use ::rsp2_structure::supercell::SupercellToken;
use ::rsp2_newtype_indices::{Idx, Indexed, cast_index};
use ::std::collections::BTreeMap;
use ::std::collections::HashMap;

use ::slice_of_array::prelude::*;

#[derive(Debug, Clone)]
pub struct ForceSets {
    num_atoms: usize,
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

// index types for local use, to aid in reasoning
newtype_index!{DispI}  // index of a displacement
newtype_index!{PrimI}  // index of a primitive cell site
newtype_index!{SuperI} // index of a supercell site
newtype_index!{StarI}  // index of a star of sites equivalent under symmetry
newtype_index!{OperI}  // index into the space group

impl ForceSets {
    pub fn zero(num_atoms: usize) -> Self {
        ForceSets {
            num_atoms,
            atom_displaced: vec![],
            atom_affected: vec![],
            cart_force: vec![],
            cart_displacement: vec![],
        }
    }

    // NOTE: This basically follows Phonopy's method to a T, although
    //       it is a bit smarter in some places, and probably dumber in other places.
    //
    //       (notably, phonopy translates the displaced atom to the origin and performs
    //        expensive brute force overlap searches, while this leaves everything where
    //        they are and adjusts indices to simulate translation by a lattice point)
    //
    // FIXME misplaced
    // FIXME break up
    pub fn like_phonopy(
        displacements: &[(usize, V3)],      // [displacement] -> (displaced, cart_disp)
        force_sets: &[BTreeMap<usize, V3>], // [displacement][affected] -> cart_force
        frac_rots: &[M33<i32>], // HACK
        cart_rots: &[M33],                  // [sg_index] -> matrix
        super_deperms: &[Perm],             // [sg_index] -> perm
        sc: &SupercellToken,
        perm_HACK_to_phonopy: &Perm,
    ) -> FailResult<ForceConstants>
    {
        let permute_atom = |deperm: &Perm, i: SuperI| {
            SuperI::new(deperm.permute_index(i.index()))
        };

        let displacements: &[(SuperI, V3)] = cast_index(displacements);
        let displacements: &Indexed<DispI, [_]> = Indexed::from_raw_ref(displacements);

        let force_sets: &[BTreeMap<SuperI, V3>] = cast_index(force_sets);
        let force_sets: &Indexed<DispI, [_]> = Indexed::from_raw_ref(force_sets);

        let cart_rots:     &Indexed<OperI, [M33]> = Indexed::from_raw_ref(cart_rots);
        let super_deperms: &Indexed<OperI, [Perm]> = Indexed::from_raw_ref(super_deperms);

        // Gather displacements for each star of sites.
        //
        // The non-equivalent atoms were discovered by phonopy and taken into account
        // in its generated displacements (always using the same atom from each star).
        // All we have to do is gather displacements with the same index.
        let displacements_by_star: Indexed<StarI, Vec<(SuperI, Vec<DispI>)>>;
        displacements_by_star = {
            // Use BTreeMap for consistent ordering, because the result of this block
            // is what defines the meaning of `StarI`.
            let mut map = BTreeMap::<SuperI, _>::new();
            for (disp_i, &(super_i, _)) in displacements.iter_enumerated() {
                map.entry(super_i)
                    .or_insert_with(Vec::new)
                    .push(disp_i)
            }
            map.into_iter().collect::<_>()
        };

        let primitive_atoms: Indexed<SuperI, Vec<PrimI>>;
        let cells:           Indexed<SuperI, Vec<[u32; 3]>>;
        let lattice_points:  Indexed<SuperI, Vec<V3<i32>>>;
        primitive_atoms = Indexed::from_raw(cast_index(sc.atom_primitive_atoms()));
        cells = Indexed::from_raw(sc.atom_cells());
        lattice_points = Indexed::from_raw(sc.atom_lattice_points());

        // Displaced atoms are always in this image of the primitive cell.
        //
        // These are the only rows of the force constants table that we actually need to compute.
        let displaced_lattice_point = sc.lattice_point_from_cell(DISPLACED_CELL);

        // Find the star of each primitive site,
        // and an oper that maps the representative atom into it.
        #[derive(Debug, Clone)]
        struct PrimStarData {
            star: StarI,
            // operators that map the representative atom into this position.
            opers_from_rep: Vec<OperI>,
        }

        let prim_star_data: Indexed<PrimI, Vec<PrimStarData>> = {
            let mut data = Indexed::<PrimI, _>::from_elem_n(None, sc.num_primitive_atoms());
            for (star, &(representative_atom, _)) in displacements_by_star.iter_enumerated() {
                for (oper, deperm) in super_deperms.iter_enumerated() {
                    let permuted_atom = permute_atom(deperm, representative_atom);
                    let prim = primitive_atoms[permuted_atom];

                    if data[prim].is_none() {
                        data[prim] = Some(PrimStarData { star, opers_from_rep: vec![] });
                    }

                    let existing = data[prim].as_mut().expect("BUG!");
                    assert_eq!(
                        star, existing.star,
                        "displacements contained multiple atoms in same symmetry star!",
                    );
                    existing.opers_from_rep.push(oper)
                }
            }
            data.into_iter().map(|p| p.expect("BUG!")).collect()
        };

        #[derive(Serialize)]
        struct JaySon(Vec<JaySonItem>);
        #[derive(Serialize)]
        struct JaySonItem {
            displaced: usize,
            displacements: Vec<JaySonDisplacement>,
            original_forces: Vec<Vec<V3>>,
            pseudoinverse: Vec<Vec<f64>>,
            forces: Vec<JaySonForce>,
        }
        #[derive(Serialize)]
        struct JaySonDisplacement {
            oper: M33<i32>,
            cartoper: M33,
            vector: V3,
        }
        #[derive(Serialize)]
        struct JaySonForce {
            affected: usize,
            vectors: Vec<V3>,
        }

        let mut jay_son = JaySon(vec![]);


        let mut row_atom = vec![];
        let mut col_atom = vec![];
        let mut cart_matrix = vec![];
        for (star, (displaced_atom, disp_indices)) in displacements_by_star.into_iter_enumerated() {
            assert_eq!(lattice_points[displaced_atom], displaced_lattice_point);

            let representative = primitive_atoms[displaced_atom];
            let mut jay_son_original_forces = vec![];

            // Expand the available data using symmetry.
            //
            // Ignore symmetry operators that map the representative to another
            // primitive site. (after we solve for the representative's force
            // constants, the others within its star are related by symmetry)
            assert_eq!(star, prim_star_data[representative].star, "BUG!");

            // Row of a table of equations that will be solved by pseudo inverse.
            newtype_index!(EqnI);

            // Inspired by phonopy, we build the complete forces in such a way that
            // ensures that the forces at each affected atom are described by the same
            // displacements (in the same order).
            // This way we only need to compute one pseudoinverse per symmetry star.
            let mut all_displacements = Indexed::<EqnI, Vec<V3>>::new();
            let mut all_sparse_forces = BTreeMap::<SuperI, BTreeMap<EqnI, V3>>::new();

            let mut jay_son_displacements = vec![];

            // (opers that map the displaced (representative) atom into an image of itself
            //  under the primitive lattice)
            for &oper in &prim_star_data[representative].opers_from_rep {
                // Picture the following:
                //
                // * Suppose that our supercell were printed out, so that we may lay a
                //   transparency sheet over it. (also suppose that the structure were 2D, so
                //   that the printout could faithfully render it without perspective)
                // * Draw arrows on all atoms representing the present forces, and draw a
                //   circle around the displaced atom.  This is the input set of data.
                // * Now apply `cart_rot` to the transparency sheet.
                //   **This is the output set of data.**

                // Notice that rotating the sheet may move the displaced atom to an
                // image of itself.  When this occurs, the resulting set of data
                // corresponds to a row of the force constants matrix that we don't
                // directly care about.
                //
                // However, by applying a lattice point translation to bring the
                // displaced atom back, we get data for the row we DO care about.
                let rotate_and_translate_atom = {
                    // "rotating a transparency" ~~ applying a depermutation
                    let rotate = |atom| permute_atom(&super_deperms[oper], atom);

                    let correction: V3<i32> = {
                        let desired = displaced_lattice_point;
                        let actual = lattice_points[rotate(displaced_atom)];
                        desired - actual
                    };

                    let primitive_atoms = &primitive_atoms;
                    let lattice_points = &lattice_points;
                    move |atom| {
                        let atom = rotate(atom);

                        // translate
                        // (this method will wrap around the supercell lattice for us)
                        SuperI::new(sc.atom_from_lattice_point(
                            primitive_atoms[atom].index(),
                            lattice_points[atom] + correction,
                        ))
                    }
                };
                let rotate_vector = |v: V3| v * cart_rots[oper].t();

                assert_eq!(
                    rotate_and_translate_atom(displaced_atom),
                    displaced_atom,
                );


                for &disp in &disp_indices {
                    assert_eq!(displacements[disp].0, displaced_atom);

                    jay_son_original_forces.push({
                        let mut out = vec![V3::zero(); sc.num_supercell_atoms()];
                        for (&key, &thing) in &force_sets[disp] {
                            out[perm_HACK_to_phonopy.permute_index(key.index())] = thing;
                        }
                        out
                    });

                    let eqn_i = {
                        let new_displacement = rotate_vector(displacements[disp].1);
                        jay_son_displacements.push(JaySonDisplacement {
                            oper: frac_rots[oper.index()],
                            cartoper: cart_rots[oper],
                            vector: new_displacement,
                        });
                        all_displacements.push(new_displacement)
                    };

                    for (&affected_atom, &cart_force) in &force_sets[disp] {
                        use ::util::ext_traits::OptionExpectNoneExt;

                        let new_affected_atom = rotate_and_translate_atom(affected_atom);
                        let new_cart_force = rotate_vector(cart_force);
                        all_sparse_forces
                            .entry(new_affected_atom)
                            .or_insert_with(BTreeMap::new)
                            .insert(eqn_i, new_cart_force)
                            .expect_none("BUG! multiple affected atoms rotated to same one?!")
                    }
                }
            }
            let all_displacements = all_displacements;
            let all_sparse_forces = all_sparse_forces;

            let jay_son_displaced = perm_HACK_to_phonopy.permute_index(displaced_atom.index());

            use rsp2_linalg::{CMatrix, dot, left_pseudoinverse};

            let num_eqns = all_displacements.len();

            // all forces we just computed use the same displacements,
            // so we only need to compute a single pseudoinverse
            let pseudoinverse: CMatrix = left_pseudoinverse((&all_displacements.raw[..]).into())?;
            let jay_son_pseudoinverse = {
                (0..pseudoinverse.rows()).map(|r| (0..pseudoinverse.cols()).map(|c| pseudoinverse[(r, c)]).collect()).collect()
            };

            let mut jay_son_forces = vec![];

            let force_constants_row: BTreeMap<SuperI, M33> = {
                all_sparse_forces.into_iter()
                    // "Densify" the BTreeMap<EqnI, V3>s into Vecs.
                    // (it is not expected that missing elements should be a common occurrence,
                    //  but in theory they *could* occur for atoms located on the order of
                    //  symmetry precision away from the cutoff radius)
                    .map(|(atom, force_map)| {
                        if force_map.len() < num_eqns {
                            // rare circumstance --> chance for bitrot.
                            // Leave a note, but don't spam.
                            info_once!("\
                                Found atoms with nonzero forces at some rotations, but not others. \
                                This is a rare circumstance!...but don't worry too much, I *think* \
                                it is handled correctly. \
                            ");
                        }

                        let mut force_vec = Indexed::from_elem(V3::zero(), &all_displacements);
                        for (eqn, v3) in force_map {
                            force_vec[eqn] = v3;
                        }
                        (atom, force_vec)
                    })
                    // solve for the fcs
                    .map(|(atom, force_vec)| {
                        jay_son_forces.push(JaySonForce { affected: perm_HACK_to_phonopy.permute_index(atom.index()), vectors: force_vec.raw.clone() });
                        let forces: CMatrix = force_vec.raw.into();
                        let phi: CMatrix = dot(&*pseudoinverse, &*forces).into();
                        let phi: M33 = -M3(phi.c_order_data().nest::<V3>().to_array());
                        (atom, phi)
                    })
                    .collect()
            };

            assert_eq!(lattice_points[displaced_atom], displaced_lattice_point);
            let (col_atom_part, cart_matrix_part): (Vec<_>, Vec<_>) = force_constants_row.into_iter().unzip();
            let row_atom_part = vec![displaced_atom; col_atom_part.len()];

            let row_atom_part: Vec<usize> = cast_index(row_atom_part);
            let col_atom_part: Vec<usize> = cast_index(col_atom_part);
            row_atom.extend(row_atom_part);
            col_atom.extend(col_atom_part);
            cart_matrix.extend(cart_matrix_part);

            jay_son.0.push(JaySonItem {
                displaced: jay_son_displaced,
                displacements: jay_son_displacements,
                original_forces: jay_son_original_forces,
                forces: jay_son_forces,
                pseudoinverse: jay_son_pseudoinverse,
            });


            // FIXME rotate to get other atoms in star
        } // for star
        trace!("{}", ::serde_json::to_string(&jay_son).unwrap());
        let num_atoms = sc.num_supercell_atoms();
        Ok(ForceConstants { num_atoms, row_atom, col_atom, cart_matrix })
    }

    pub fn from_displacement(
        num_atoms: usize,
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
        ForceSets { num_atoms, atom_affected, atom_displaced, cart_force, cart_displacement }
    }


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

        let num_atoms = self.num_atoms;
        ForceSets { num_atoms, atom_displaced, atom_affected, cart_force, cart_displacement }
    }

    // FIXME shouldn't be pub(crate)
    pub(crate) fn concat_from<Ss>(iter: Ss) -> Option<Self>
    where Ss: IntoIterator<Item=ForceSets>,
    {
        use ::itertools::Itertools;
        iter.into_iter().fold1(|mut a, b| {
            assert_eq!(a.num_atoms, b.num_atoms);
            a.atom_affected.extend(b.atom_affected);
            a.atom_displaced.extend(b.atom_displaced);
            a.cart_force.extend(b.cart_force);
            a.cart_displacement.extend(b.cart_displacement);
            a
        })
    }

    // FIXME shouldn't be pub(crate)
    pub(crate) fn solve_force_constants(&self, sc: &SupercellToken
                                        , perm_HACK: &Perm
        , carts_HACK: &[V3]
    ) -> FailResult<ForceConstants>
    {
        use ::util::zip_eq as z;
        let ForceSets {
            num_atoms,
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

        let fcs = {
            ForceConstants { num_atoms, row_atom, col_atom, cart_matrix }
                .add_rows_for_other_cells(sc)
                //.symmetrize_by_transpose()
                .impose_perm_and_translational_invariance_a_la_phonopy()
                .canonicalize()
        };
        Ok(fcs)
    }
}

#[derive(Debug, Clone)]
pub struct ForceConstants { // conceptually ForceConstants<I: Idx>
    num_atoms: usize,
    row_atom: Vec<usize>,   // conceptually Vec<I>
    col_atom: Vec<usize>,   // conceptually Vec<I>
    cart_matrix: Vec<M33>,
}

impl ForceConstants {
    // take ForceConstants where row_atom is always in DISPLACED_CELL
    // and generate all the other rows
    fn add_rows_for_other_cells(mut self, sc: &SupercellToken) -> Self {
        assert!({
            let cells = sc.atom_cells();
            self.row_atom.iter().all(|&row| cells[row] == DISPLACED_CELL)
        });

        let old_len = self.row_atom.len();
        for axis in 0..3 {
            // get deperm that translates data by one cell along this axis
            let unit = V3::from_fn(|i| (i == axis) as i32);
            let deperm = sc.lattice_point_translation_deperm(unit);

            let mut permuted_fcs = self.clone();

            // skip 0 because 'self' already has the data for 0 cell translation
            for _ in 1..sc.periods()[axis] {
                permuted_fcs = permuted_fcs.permuted_by(&deperm);
                self.add(permuted_fcs.clone());
            }
        }
        assert_eq!(self.row_atom.len(), old_len * sc.num_cells());
        self
    }

    fn symmetrize_by_transpose(mut self) -> Self {
        let t = self.clone().transpose();
        self.add(t);
        for m in &mut self.cart_matrix {
            *m *= 0.5;
        }
        self
    }

    /// Transpose as if it were a 3N by 3N matrix.
    fn transpose(self) -> Self {
        let ForceConstants { num_atoms, row_atom, col_atom, cart_matrix } = self;
        let cart_matrix = cart_matrix.into_iter().map(|m| m.t()).collect();
        ForceConstants {
            num_atoms,
            row_atom: col_atom,
            col_atom: row_atom,
            cart_matrix,
        }
    }

    fn add(&mut self, other: Self) {
        assert_eq!(self.num_atoms, other.num_atoms);
        self.row_atom.extend(other.row_atom);
        self.col_atom.extend(other.col_atom);
        self.cart_matrix.extend(other.cart_matrix);
    }

//    // this is something done by the C code in Phonopy which purportedly
//    // imposes translational invariance.
//    fn impose_translational_invariance_a_la_phonopy(mut self) -> Self {
//
//        // first, we need to iterate over each row, which is unnatural for our COO-based
//        // representation (that's our fault)
//        fn insertion_index<K: Ord>(data: &[K], value: K) -> usize {
//            match data.binary_search(&value) {
//                Ok(i) | Err(i) => i,
//            }
//        }
//
//        self.sort_data_by(|r, _| r);
//        let row_ptr = (0..=self.num_atoms).map(|r| insertion_index(&self.row_atom, r)).collect::<Vec<_>>();
//        for &[row_start, row_end] in row_ptr.windows(2) {
//            // okay.  Here's where things get interesting.
//            //
//            // For each of the 3 diagonal matrix components labeled by `(r_k, r_k)`,
//            // phonopy's C code sums up all values in that component along the row for atom `r_atom`,
//            // and subtracts it from `[r_atom][r_atom][r_k][r_k]`.
//            //
//            // For each of the 3 off-diagonal components labeled by `(r_k, c_k)` where `c_k > r_k`,
//            // the code sums up all values in that component along the row for atom `r_atom`,
//            // and subtracts it from two elements in the `[r_atom][r_atom]` matrix (`[r_k][c_k]`
//            // and `[c_k][r_k]`).
//            //
//            // If you look at the original python code, you'll see that it used to do something
//            // that first went over each row, then over each column.  ISTM that the "over each column"
//            // part is why the `[c_k][r_k]` term is modified, and that this is equivalent presuming that
//            // the
//            // Notice that there are data dependencies here.
//            //
//            //
//            fn without_diagonal(mut mat: M33) -> M33 {
//                for k in 0..3 {
//                    mat[k][k] = 0;
//                }
//                mat
//            }
//            let start =
//        }
//    }

    // this is something done by the C code in Phonopy which purportedly
    // imposes translational invariance.
    fn impose_perm_and_translational_invariance_a_la_phonopy(mut self) -> Self {
        let mut dense = self.to_dense_matrix();

        for i in 0..dense.len() {
            let rest = &mut dense[i..];
            let (row_i, rest) = rest.split_first_mut().unwrap();

            /* non diagonal part */
            for (offset_j, row_j) in rest.iter_mut().enumerate() {
                let j = offset_j + i + 1;
                for k in 0..3 {
                    for l in 0..3 {
                        let elem_m = &mut row_i[j][k][l];
                        let elem_n = &mut row_j[i][l][k];
                        *elem_m += *elem_n;
                        *elem_m /= 2.0;
                        *elem_n = *elem_m;
                    }
                }
            }

            /* diagnoal part */
            let diag = &mut row_i[i];
            for k in 1..3 {
                for l in k + 1..3 {
                    diag[k][l] += diag[l][k];
                    diag[k][l] /= 2.0;
                    diag[l][k] = diag[k][l];
                }
            }
        }

        for i in 0..dense.len() {
            for k in 0..3 {
                for l in k..3 {
                    let sum: f64 = (0..dense.len()).map(|j| dense[i][j][k][l]).sum();
                    dense[i][i][k][l] -= sum;
                    if k != l {
                        dense[i][i][l][k] -= sum;
                    }
                }
            }
        }

        Self::from_dense_matrix(dense)
    }

    // HACK
    fn from_dense_matrix(mat: Vec<Vec<[[f64; 3]; 3]>>) -> ForceConstants {
        let num_atoms = mat.len();
        let (mut row_atom, mut col_atom, mut cart_matrix) = (vec![], vec![], vec![]);
        for (r, row) in mat.into_iter().enumerate() {
            for (c, m) in row.into_iter().enumerate() {
                let m = mat::from_array(m);
                if m != M33::zero() {
                    row_atom.push(r);
                    col_atom.push(c);
                    cart_matrix.push(m);
                }
            }
        }
        ForceConstants { num_atoms, row_atom, col_atom, cart_matrix }
    }


    fn sort_data_by<K, F>(self, mut f: F) -> Self
    where
        K: Ord,
        F: FnMut(usize, usize) -> K,
    {
        let ForceConstants { num_atoms, row_atom, col_atom, cart_matrix } = self;
        let data = zip_eq!(&row_atom, &col_atom).map(|(&r, &c)| f(r, c)).collect::<Vec<_>>();
        let perm = Perm::argsort(&data);

        let row_atom = row_atom.permuted_by(&perm);
        let col_atom = col_atom.permuted_by(&perm);
        let cart_matrix = cart_matrix.permuted_by(&perm);
        ForceConstants { num_atoms, row_atom, col_atom, cart_matrix }
    }

    /// Ensure that at most a single item per atom pair exists by summing duplicates.
    fn canonicalize(self) -> Self {
        let row_major = self.sort_data_by(|r, c| (r, c));
        let ForceConstants { num_atoms, row_atom, col_atom, cart_matrix } = row_major;

        // reduce
        let iter = zip_eq!(zip_eq!(row_atom, col_atom), cart_matrix);
        let iter = IntoReducedIter::new(iter, ::std::ops::Add::add);
        let (pos, cart_matrix): (Vec<_>, _) = iter.unzip();
        let (row_atom, col_atom) = pos.into_iter().unzip();

        ForceConstants { num_atoms, row_atom, col_atom, cart_matrix }
    }
}

// both the rows and columns of ForceConstants are conceptually indexed
// by the same index type, so the Permute impl permutes both.
impl Permute for ForceConstants {
    fn permuted_by(self, perm: &Perm) -> ForceConstants {
        let ForceConstants { num_atoms, mut row_atom, mut col_atom, cart_matrix } = self;
        for row in &mut row_atom {
            *row = perm.permute_index(*row);
        }
        for col in &mut col_atom {
            *col = perm.permute_index(*col);
        }
        ForceConstants { num_atoms, row_atom, col_atom, cart_matrix }
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
    pub(crate) fn to_dense_matrix(&self) -> Vec<Vec<[[f64; 3]; 3]>> {
        use ::rsp2_array_types::Unvee;
        let n_basis = self.num_atoms;
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

