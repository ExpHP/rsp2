/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

#[macro_use] extern crate log;
#[macro_use] extern crate serde;
#[macro_use] extern crate failure;
#[macro_use] extern crate rsp2_newtype_indices;
#[macro_use] extern crate rsp2_util_macros;

use rsp2_array_types::{V3, M33, M3};
use rsp2_soa_ops::{Perm, Permute};
use rsp2_structure::{Coords};
use rsp2_structure::supercell::SupercellToken;
use rsp2_newtype_indices::{Idx, Indexed, index_cast};
use rsp2_sparse::{RawBee, RawCoo, RawCsr};
use std::collections::{BTreeMap, BTreeSet};
use slice_of_array::prelude::*;

pub type FailResult<T> = Result<T, failure::Error>;

// index types for local use, to aid in reasoning
newtype_index!{DispI}  // index of a displacement
newtype_index!{PrimI}  // index of a primitive cell site
newtype_index!{SuperI} // index of a supercell site
newtype_index!{StarI}  // index of a star of sites equivalent under symmetry
newtype_index!{OperI}  // index into the space group
newtype_index!{EqnI}   // row of a table of equations that will be solved by pseudo inverse

/// Capable of representing the complete force constants matrix, including redundant rows that are
/// related by translation to other rows.  Only really used for debugging and for tests.
///
/// Does not *necessarily* contain data at all rows.  That is, some instances of
/// `SuperForceConstants` might only have the the rows in `ForceConstants::DESIGNATED_CELL` filled.
/// (or perhaps it has some other arbitrary set of indices filled, such as e.g. the indices chosen
/// to be displaced by another tool such as phonopy)
#[derive(Debug, Clone)]
pub struct SuperForceConstants(
    RawCoo<M33, SuperI, SuperI>,
);

/// Force constants matrix, with only rows for atoms in `ForceConstants::DESIGNATED_CELL` stored.
#[derive(Debug, Clone)]
pub struct ForceConstants(
    pub RawCsr<M33, PrimI, SuperI>,
);

#[derive(Debug, Clone)]
pub struct DynamicalMatrix(
    pub RawCsr<Complex33, PrimI, PrimI>,
);

const DESIGNATED_CELL: [u32; 3] = [0, 0, 0];

impl ForceConstants {
    /// Displaced atoms must always be in this cell.
    pub const DESIGNATED_CELL: [u32; 3] = DESIGNATED_CELL;

    /// Compute a partially filled force constants matrix that contains only the rows
    /// that appear in the definition of the dynamical matrix. (those where the displaced
    /// atom is an image in `ForceConstants::DESIGNATED_CELL`)
    ///
    /// This basically follows Phonopy's method for `--writefc` to a T, although it
    /// is a bit smarter in some places, and probably dumber in other places.
    ///
    /// Notable differences from phonopy include:
    ///
    /// * This uses a sparse format for anything indexed by atoms (primitive or super)
    /// * Phonopy translates each displaced atom to the origin and performs
    ///   expensive brute force overlap searches, while this leaves everything where
    ///   they are and adjusts indices to simulate translation by a lattice point)
    pub fn compute_required_rows(
        // Displacements, using the same supercell indices that the corresponding
        // `force_sets` were generated from.  Currently these are all required to use
        // the images in `ForceConstants::DESIGNATED_CELL`. (this requirement is checked)
        //
        // Furthermore, for each symmetry star of sites in the primitive cell, only one of those
        // sites may appear in this list. (this requirement is checked)
        super_displacements: &[(usize, V3)], // [displacement] -> (super_displaced, cart_disp)
        force_sets: &[BTreeMap<usize, V3>],  // [displacement][affected] -> cart_force
        cart_rots: &[M33],                   // [sg_index] -> matrix
        super_deperms: &[Perm],              // [sg_index] -> permutation on supercell
        sc: &SupercellToken,
    ) -> FailResult<ForceConstants>
    {
        // wrap data with information about index type
        //
        // most type annotations in here are not strictly necessary, but serve as a stop-gap measure
        // to ensure that at least *something* close to the public interface stops compiling if the
        // newtyped indices in Context are changed (to remind you to check callers)
        let super_displacements: &[(SuperI, V3)] = index_cast(super_displacements);
        let super_displacements: &Indexed<DispI, [_]> = Indexed::from_raw_ref(super_displacements);

        let force_sets: &[BTreeMap<SuperI, V3>] = index_cast(force_sets);
        let force_sets: &Indexed<DispI, [_]> = Indexed::from_raw_ref(force_sets);

        let cart_rots: &Indexed<OperI, [M33]> = Indexed::from_raw_ref(cart_rots);
        let super_deperms: &Indexed<OperI, [Perm]> = Indexed::from_raw_ref(super_deperms);

        let sc = SupercellWrapper::new(sc);

        let primitive_atoms: Indexed<SuperI, Vec<PrimI>> = sc.atom_primitive_atoms();
        let lattice_points: Indexed<SuperI, Vec<V3<i32>>> = sc.atom_lattice_points();
        let primitive_atoms = &primitive_atoms[..];
        let lattice_points = &lattice_points[..];

        let displacements: Indexed<DispI, Vec<(PrimI, V3)>> = {
            super_displacements.iter()
                .map(|&(super_disp, force)| {
                    if lattice_points[super_disp] != sc.designated_lattice_point {
                        // (NOTE: this requirement could easily be lifted by correcting the indices in
                        //        force sets by applying lattice point translations that move the displaced
                        //        atom to DESIGNATED_CELL.  I have not implemented this because I haven't
                        //        needed it.)
                        bail!(
                            "\
                                ForceConstants currently requires input forces to use displaced atoms in \
                                ForceConstants::DESIGNATED_CELL (= {:?}). (found atom with cell {:?})\
                            ",
                            ForceConstants::DESIGNATED_CELL,
                            sc.raw.cell_from_lattice_point(lattice_points[super_disp]),
                        )
                    }
                    Ok((primitive_atoms[super_disp], force))
                }).collect::<FailResult<_>>()?
        };

        Context {
            displacements: &displacements,
            sc, primitive_atoms, lattice_points, force_sets,
            cart_rots, super_deperms,
        }.compute_force_constants()
    }
}

// a variety of precomputed things made available during the whole computation
struct Context<'ctx> {
    // Function arguments
    displacements:   &'ctx Indexed<DispI, [(PrimI, V3)]>,
    force_sets:      &'ctx Indexed<DispI, [BTreeMap<SuperI, V3>]>,
    cart_rots:       &'ctx Indexed<OperI, [M33]>,
    super_deperms:   &'ctx Indexed<OperI, [Perm]>,
    sc:              SupercellWrapper<'ctx>,

    // Some data from `sc`
    primitive_atoms: &'ctx Indexed<SuperI, [PrimI]>,
    lattice_points:  &'ctx Indexed<SuperI, [V3<i32>]>,
}

impl<'ctx> Context<'ctx> {
    fn compute_force_constants(
        &self,
    ) -> FailResult<ForceConstants> {
        let (star_data, prim_data) = self.compute_symmetry_info();

        let representative_rows = self.compute_representative_rows(&star_data, &prim_data)?;

        let all_rows = self.derive_rows_by_symmetry(&star_data, &prim_data, representative_rows);

        let matrix = {
            let dim = (self.sc.raw.num_supercell_atoms(), self.sc.raw.num_supercell_atoms());
            let map = all_rows.into_iter_enumerated().collect();
            RawBee { dim, map }.into_csr()
        };
        Ok(ForceConstants(matrix))
    }
}

//--------------------------------------------------
// data computed early on

// ad-hoc struct with data that is computed for each symmetry star
#[derive(Debug, Clone)]
struct StarData {
    representative: PrimI,
    // operators that map the representative atom into this position.
    displacements: Vec<DispI>,
}

// ad-hoc struct relating each primitive atom to its site-symmetry representative.
#[derive(Debug, Clone)]
struct PrimData {
    star: StarI,
    // operators that move the star's representative into this primitive site.
    opers_from_rep: Vec<OperI>,
}

impl<'ctx> Context<'ctx> {
    // FIXME: It'd be nice if this used the new `Stars` type, but then the trouble is that
    //        the indices used in the displacements might not be the same indices chosen as
    //        representatives by `Stars`.
    fn compute_symmetry_info(&self) -> (Indexed<StarI, Vec<StarData>>, Indexed<PrimI, Vec<PrimData>>) {

        // Gather displacements for each star of sites.
        //
        // The non-equivalent atoms were discovered by phonopy and taken into account
        // in its generated displacements (always using the same atom from each star).
        // All we have to do is gather displacements with the same index.
        //
        // (NOTE: At this point it is assumed that the input displacements only
        //  ever displace one representative atom from each star, so that every
        //  unique index in `displacements` corresponds to a star.
        //  If this turns out not to be true, it will be caught later.)
        let star_data: Indexed<StarI, Vec<StarData>>;
        star_data = {
            // Use BTreeMap for consistent ordering, because the result of this block
            // is what defines the meaning of `StarI`.
            let mut map = BTreeMap::<PrimI, _>::new();
            for (disp_i, &(prim_i, _)) in self.displacements.iter_enumerated() {
                map.entry(prim_i)
                    .or_insert_with(Vec::new)
                    .push(disp_i)
            }
            map.into_iter()
                .map(|(representative, displacements)| StarData { representative, displacements })
                .collect()
        };

        // Gather data relating each primitive atom to its site-symmetry representative.
        let prim_data: Indexed<PrimI, Vec<PrimData>> = {
            let mut data = Indexed::<PrimI, _>::from_elem_n(None, self.sc.raw.num_primitive_atoms());
            for (star, star_data) in star_data.iter_enumerated() {
                let representative_atom = self.sc.atom_from_lattice_point(star_data.representative, self.sc.designated_lattice_point);
                for oper in self.oper_indices() {
                    let permuted_atom = self.rotate_atom(oper, representative_atom);
                    let prim = self.primitive_atoms[permuted_atom];

                    if data[prim].is_none() {
                        data[prim] = Some(PrimData { star, opers_from_rep: vec![] });
                    }

                    let existing = data[prim].as_mut().expect("BUG!");
                    assert_eq!(
                        star, existing.star,
                        "displacements contained multiple atoms in same symmetry star!",
                    );
                    existing.opers_from_rep.push(oper)
                }
            }
            data.into_iter_enumerated().map(|(prim, d)| d.unwrap_or_else(|| {
                panic!(
                    "\
                        None of the displaced sites are equivalent under symmetry \
                        to primitive site {}!!\
                    ", prim,
                );
            })).collect()
        };
        (star_data, prim_data)
    }

    // Computes the rows of the FC matrix belonging to symmetry star representatives.
    fn compute_representative_rows(
        &self,
        star_data: &Indexed<StarI, [StarData]>,
        prim_data: &Indexed<PrimI, [PrimData]>,
    ) -> FailResult<BTreeMap<PrimI, BTreeMap<SuperI, M3<V3<f64>>>>> {
        let mut computed_rows: BTreeMap<PrimI, BTreeMap<SuperI, M33>> = Default::default();
        for (star, data) in star_data.iter_enumerated() {
            let StarData {
                representative,
                displacements: ref disp_indices,
            } = *data;

            // Expand the available data using symmetry to ensure we have enough
            // independent equations for pseudoinversion.
            //
            // Ignore symmetry operators that map the representative to another
            // primitive site. (after we solve for the representative's force
            // constants, the others within its star are related by symmetry)
            assert_eq!(star, prim_data[representative].star, "BUG!");

            let (
                row_displacements,
                row_forces,
            ) = self.build_all_equations_for_representative_row(
                representative,
                &disp_indices,
                &prim_data[representative].opers_from_rep,
            );

            use rsp2_linalg::{CMatrix, dot, left_pseudoinverse};

            // all forces we just computed use the same displacements,
            // so we only need to compute a single pseudoinverse
            let pseudoinverse: CMatrix = left_pseudoinverse((&row_displacements.raw[..]).into())?;

            let force_constants_row: BTreeMap<SuperI, M33> = {
                row_forces.into_iter()
                    // solve for the fcs
                    .map(|(atom, force_vec)| {
                        let forces: CMatrix = force_vec.raw.into();
                        let phi: CMatrix = dot(&*pseudoinverse, &*forces).into();
                        let phi: M33 = -M3(phi.c_order_data().nest::<V3>().to_array());
                        (atom, phi)
                    })
                    .collect()
            };

            if let Some(_) = computed_rows.insert(representative, force_constants_row) {
                panic!("(BUG) computed same row of FCs twice!?");
            }
        } // for star
        Ok(computed_rows)
    }

    /// Use symmetry to expand the number of displacement-force equations for
    /// a row belonging to a symmetry star representative.
    //
    // Inspired by phonopy, we build the complete forces in such a way that
    // ensures that the forces at each affected atom are described by the same
    // displacements (in the same order). This way we only need to compute one
    // pseudoinverse per symmetry star.
    //
    // I think this function vaguely corresponds to the first half of phonopy's
    // `_solve_force_constants_svd` (though to be entirely honest, I just picked
    // a sizable looking block of code and did `Extract Method`)
    fn build_all_equations_for_representative_row(
        &self,
        // The row we are generating, which should be the representative atom
        // of a site-symmetry star. (needed to determine lattice point corrections
        // for each operator)
        displaced_prim: PrimI,
        // All displacements that displace this atom.
        disp_indices: &[DispI],
        // Operators that map `displaced_prim` to itself in the primitive structure.
        invariant_opers: &[OperI],
    ) -> (
        Indexed<EqnI, Vec<V3>>,
        BTreeMap<SuperI, Indexed<EqnI, Vec<V3>>>,
    ) {
        assert!(
            disp_indices.len() <= 6,
            "(BUG) got {} displacements. That's a lot! Are you sure these are all for the same atom?",
            disp_indices.len(),
        );

        let mut all_displacements = Indexed::<EqnI, Vec<V3>>::new();
        // Initially a BTreeMap<EqnI, _> is used for forces in case the set of affected atoms in
        // a force set is somehow not symmetric (in which case some rotations will have a different
        // set of affected atoms than others)
        let mut all_sparse_forces = BTreeMap::<SuperI, BTreeMap<EqnI, V3>>::new();

        for &oper in invariant_opers {

            // get a composite rotation + lattice-point translation that maps the
            // displaced supercell atom directly into itself (not just an image)
            let (rotated_prim, rotate_and_translate_atom) = self.get_corrected_rotate(oper, displaced_prim);
            assert_eq!(rotated_prim, displaced_prim, "(BUG) prim not invariant under supplied oper!");

            let rotate_vector = |v: V3| v * self.cart_rots[oper].t();

            for &disp in disp_indices {
                assert_eq!(self.displacements[disp].0, displaced_prim, "(BUG) disp for wrong atom");

                let eqn_i = all_displacements.push(rotate_vector(self.displacements[disp].1));

                for (&affected_atom, &cart_force) in &self.force_sets[disp] {
                    if cart_force != cart_force {
                        panic!("Force sets contain NaN!");
                    }
                    let new_affected_atom = rotate_and_translate_atom(affected_atom);
                    let new_cart_force = rotate_vector(cart_force);

                    if let Some(_) = {
                        all_sparse_forces
                            .entry(new_affected_atom)
                            .or_insert_with(BTreeMap::new)
                            .insert(eqn_i, new_cart_force)
                    } {
                        panic!("BUG! multiple affected atoms rotated to same one?!");
                    }
                }
            }
        }
        // done mutating
        let all_displacements = all_displacements;
        let all_sparse_forces = all_sparse_forces;

        // now that all equations have been built, we can "densify" the force matrices.
        // (with respect to equations; not the entire supercell, of course!)
        let all_forces = {
            all_sparse_forces.into_iter()
                .map(|(atom, force_map)| {
                    let mut force_vec = Indexed::from_elem(V3::zero(), &all_displacements);
                    for (eqn, v3) in force_map {
                        force_vec[eqn] = v3;
                    }
                    (atom, force_vec)
                }).collect()
        };
        (all_displacements, all_forces)
    }

    // Input: Rows for each symmetry-star representative.
    // Output: Rows for every primitive atom.
    fn derive_rows_by_symmetry(
        &self,
        star_data: &Indexed<StarI, [StarData]>,
        prim_data: &Indexed<PrimI, [PrimData]>,
        computed_rows: BTreeMap<PrimI, BTreeMap<SuperI, M3<V3<f64>>>>,
    ) -> Indexed<PrimI, Vec<BTreeMap<SuperI, M3<V3<f64>>>>>
    {
        prim_data.iter_enumerated().map(|(prim, data)| {
            let &PrimData { star, ref opers_from_rep } = data;
            let representative = star_data[star].representative;
            if computed_rows.contains_key(&prim) {
                assert_eq!(prim, representative, "(BUG!) bad input to derive_rows_by_symmetry");
                computed_rows[&prim].clone()
            } else {
                assert_ne!(prim, representative, "(BUG!) bad input to derive_rows_by_symmetry");

                // any operator will do; all should produce the same data.
                //
                // TODO: maybe we should verify that the 3x3 FC matrix is (approximately)
                //       invariant under the operations that map the site into itself?
                let &oper = opers_from_rep.get(0).expect("(BUG!) was checked earlier");

                // Visualize the operation described in the documentation of `get_corrected_rotate`.
                //
                // Performing this operation on a set of data would permute all indices according to the
                // deperm returned by `get_corrected_rotate`, and would rotate all displacements and forces
                // by the rotation part of the operator.  How does this impact the force constants?
                //
                // Clearly, the columns will be permuted by the deperm:
                let (rotated_prim, apply_deperm) = self.get_corrected_rotate(oper, representative);
                assert_eq!(rotated_prim, prim, "(BUG) oper produced different row?!");

                // But their contents will also be transformed by the rotation.
                //
                // Consider that the new force and displacement are each obtained by rotating the old
                // vectors by R, and substitute into the force constants equation. (`F = - U Phi`,
                // recalling that rsp2 prefers a row-vector-centric formalism for everything except
                // spatial transformation operators like R)
                //
                // The result:      Phi_new = R.T.inv() Phi_old R.T
                // or equivalently: Phi_new = R Phi_old R.T
                let cart_rot = self.cart_rots[oper];

                computed_rows[&representative].iter()
                    .map(|(&affected, fc_matrix)| {
                        let affected = apply_deperm(affected);
                        // FIXME: This could use a test where cart_rot != cart_rot.t()
                        // (for AB blg, cart_rot is currently always able to be E or i)
                        let fc_matrix = cart_rot * fc_matrix * cart_rot.t();
                        (affected, fc_matrix)
                    })
                    .collect()
            }
        }).collect()
    }

    // Get a depermutation that applies a spacegroup operator to force sets
    // in a way that derives new, meaningful data. The depermutation
    // is given in the form of a function that maps sparse indices.
    //
    // ---
    //
    // Picture the following:
    //
    // * Suppose that our supercell were printed out, so that we may lay a
    //   transparency sheet over it. (also suppose that the structure were 2D, so
    //   that the printout could faithfully render it without perspective).
    // * Draw arrows on various atoms representing forces. Draw a circle around the
    //   displaced atom, and a displacement vector originating from it.
    //   This is the initial set of data.
    // * Now apply a spacegroup operator to the transparency sheet.  The circle
    //   will move to a new site, and the arrows will move to new sites and rotate.
    //   This is a new set of data.
    //
    // Now, there's a catch; we only care about the rows of the force constants table that
    // correspond to the displaced atom lying in a specific image of primitive cell
    // (`designated_lattice_point`). Rotating the sheet may have moved the displaced atom
    // outside of this cell.
    //
    // However, by following up with a uniform lattice point translation, we can bring the
    // displaced atom back to the correct cell, and end up with data for a row that we DO
    // care about.
    //
    // The returned depermutation simulates this sequence of a rotation followed by a
    // corrective lattice point translation. Also returned is the new index of the displaced
    // atom.
    fn get_corrected_rotate<'a>(&'a self, oper: OperI, displaced: PrimI) -> (PrimI, impl Fn(SuperI) -> SuperI + 'a)
    {
        let displaced_super = self.sc.designated_super(displaced);

        let correction: V3<i32> = {
            let desired = self.sc.designated_lattice_point;
            let actual = self.lattice_points[self.rotate_atom(oper, displaced_super)];
            desired - actual
        };

        let rotate_and_translate = move |atom| {
            let atom = self.rotate_atom(oper, atom);
            self.sc.atom_from_lattice_point(
                self.primitive_atoms[atom],
                self.lattice_points[atom] + correction,
            )
        };

        let final_displaced_super = rotate_and_translate(displaced_super);
        let final_displaced = self.prim_from_designated_super(final_displaced_super);
        (final_displaced, rotate_and_translate)
    }

    // --------------
    // helpers that wrap methods with newtyped indices

    fn oper_indices(&self) -> impl Iterator<Item=OperI> { self.super_deperms.indices() }

    // depermutations in the form of a function that maps sparse indices
    fn rotate_atom(&self, oper: OperI, atom: SuperI) -> SuperI {
        let deperm = &self.super_deperms[oper];
        SuperI::new(deperm.permute_index(atom.index()))
    }

    // recovers the primitive index from a supercell index whose lattice point is
    // `designated_lattice_point`. (panics if this does not hold, in which case you
    // may have forgotten to apply a translation or something)
    fn prim_from_designated_super(&self, atom: SuperI) -> PrimI {
        assert_eq!(self.sc.designated_lattice_point, self.lattice_points[atom]);
        self.primitive_atoms[atom]
    }
}

impl ForceConstants {
    /// Compute the dynamical matrix at a q-point.
    ///
    /// The force constants do not need to contain data for rows outside the
    /// designated cell. (but if they do, it won't hurt)
    pub fn dynmat_at_cart_q(
        &self,
        super_coords: &Coords,
        qpoint_cart: V3,
        sc: &SupercellToken,
        masses: &[f64],
    ) -> DynamicalMatrix {
        assert_eq!(masses.len(), sc.num_primitive_atoms());
        let masses: &Indexed<PrimI, _> = Indexed::from_raw_ref(&masses[..]);

        let sc = SupercellWrapper::new(sc);
        let primitive_atoms = sc.atom_primitive_atoms();

        // Since rsp2 pays so much attention to image vectors (see FracBond) in places,
        // one might wonder why it is now doing brute-force searches for nearest images.
        //
        // Well... our current representation of force_sets doesn't contain image information.
        // All we know is that our supercell should be large enough that each site in the
        // designated cell only interacts with one image of any other site in the supercell.
        let image_finder = rsp2_structure::NearestImageFinder::new(super_coords.lattice()).unwrap();

        // Reducing the carts is a requirement for NearestImageFinder::shortest_images_cart_fast
        let reduced_carts = Indexed::<SuperI, _>::from_raw({
            let mut reduced = super_coords.clone();
            reduced.reduce_positions();
            reduced.to_carts()
        });

        let mut shortest_images_buf = vec![];
        let coo = self.0.to_coo();
        let iter = zip_eq!(&coo.row, &coo.col, &coo.val)
            // each column of the dynamical matrix sums over columns for images in
            // the force constants matrix, with phase factors.
            .map(|(&prim_r, &super_c, &mat)| {
                let super_r = sc.designated_super(prim_r);
                let prim_c = primitive_atoms[super_c];

                // mass-normalizing scale factor
                let scale = 1.0 / f64::sqrt(masses[prim_r] * masses[prim_c]);

                // vector from atom in primitive cell to atom in supercell.
                //
                // NOTE: Strictly speaking, rsp2 does not currently need to consider ties for
                // length. If necessary, we can optimize this by only getting one vector.
                //
                // I believe that the reason that phonopy must do this is because phonopy's
                // forces are dense; phonopy assumes that the equilibrium structure has zero
                // force, so the small forces that are present in the equilibrium structure
                // will also appear between distant, noninteracting sites in the force sets.
                // Considering ties thus helps cancel out some terms that would otherwise
                // violate symmetry.
                //
                // In rsp2, noninteracting sites in the supercell truly have zero force
                // between them. There should not be any ties for any of the `(r, c)` pairs
                // that appear in our sparse force constants.
                //
                // For now, however, we *do* find all shortest images, simply because I don't want
                // this to come back to bite me if things change in the future.

                // `_fast` is okay because the carts are reduced.
                image_finder.shortest_images_cart_fast(
                    &mut shortest_images_buf,
                    reduced_carts[super_c] - reduced_carts[super_r],
                    1e-4,
                );
                if shortest_images_buf.len() > 1 {
                    // If this warning is generated spuriously, then it probably means a potential
                    // was added to rsp2 that is incapable of producing sparse force sets.
                    //
                    // If that is indeed the case, the existing code should still handle it properly
                    // (though it hasn't been tested under these conditions), and either the
                    // potential should be fixed if possible or the warning should be removed.
                    warn_once!("\
                        Multiple shortest images found for a vector in the force constants! \
                        This could mean that your supercell is not large enough.\n\
                        (or, the force sets have somehow become dense!)\n\
                        Lattice: {:?}\n\
                        Vectors: {:?}\n\
                          Force: {:?}\n\
                    ", super_coords.lattice(), shortest_images_buf, mat);
                }

                let (phase_real, phase_imag) = {
                    shortest_images_buf.iter()
                        .map(|cart_diff| V3::dot(&qpoint_cart, cart_diff))
                        .map(|arg| 2.0 * std::f64::consts::PI * arg)
                        .map(|arg| (arg.cos(), arg.sin()))
                        .fold((0.0, 0.0), |(ar, ai), (br, bi)| (ar + br, ai + bi))
                };

                // NOTE: dividing by the multiplicity here is something that phonopy does.
                //       I'm not sure precisely *why* it does so. (simply summing over the images
                //       seems more natural than averaging over them).  But for our sanity,
                //       we'll do what they do.
                //
                //       (I don't think it matters much anyways in the end, because multiplicity
                //        should only ever exceed 1 for distant atoms (about half of the supercell),
                //        so the forces are small to begin with)
                //
                //       Notably, this also makes the graphene_denseforce_111 test succeed...
                //       though I am still suspicious as to how that test is even correct!
                let real = scale * phase_real * mat / shortest_images_buf.len() as f64;
                let imag = scale * phase_imag * mat / shortest_images_buf.len() as f64;

                ((prim_r, prim_c), Complex33(real, imag))
            });

        let matrix = {
            let (pos, val): (Vec<_>, Vec<_>) = iter.unzip();
            let (row, col) = pos.into_iter().unzip();
            let dim = (sc.raw.num_primitive_atoms(), sc.raw.num_primitive_atoms());

            if val != val {
                panic!("Dynamical matrix contains NaN!");
            }
            RawCoo { dim, val, row, col }.into_csr()
        };

        DynamicalMatrix(matrix)
    }
}

// ------------------------------------------------------

impl ForceConstants {
    /// Convert to a type capable of containing data in arbitrary supercell rows, but do not yet
    /// actually fill the rows outside `ForceConstants::DESIGNATED_CELL`. (the data will still be
    /// sparse; this mostly just replaces primitive indices with supercell indices)
    ///
    /// Why?  Well, it provides a `Permute` impl, which could be used to construct something closer
    /// to the output of another tool like phonopy.  It also provides conversions to and from dense
    /// matrices, for debugging and testing.
    pub fn to_super_force_constants_with_zeroed_rows(&self, sc: &SupercellToken) -> SuperForceConstants {
        let sc = SupercellWrapper::new(sc);
        SuperForceConstants({
            self.0.to_coo()
                .map_row_indices(
                    sc.raw.num_supercell_atoms(),
                    |prim| sc.designated_super(prim),
                )
        })
    }

    pub fn to_super_force_constants_with_all_rows(&self, sc: &SupercellToken) -> SuperForceConstants {
        self.to_super_force_constants_with_zeroed_rows(sc)
            .add_rows_for_other_cells(sc)
    }
}

impl SuperForceConstants {
    /// Take `SuperForceConstants` where `row_atom` is always in `DISPLACED_CELL`
    /// and generate all the other rows.
    ///
    /// Useful for unit tests.
    pub fn add_rows_for_other_cells(mut self, sc: &SupercellToken) -> SuperForceConstants {
        let sc = SupercellWrapper::new(sc);

        assert!({
            let cells = sc.atom_cells();
            self.0.row.iter().all(|&row| cells[row] == DESIGNATED_CELL)
        });

        let old_len = self.0.row.len();
        for axis in 0..3 {
            // get deperm that translates data by one cell along this axis
            let unit = V3::from_fn(|i| (i == axis) as i32);
            let deperm = sc.raw.lattice_point_translation_deperm(unit);

            let mut permuted_fcs = self.clone();

            // skip 0 because 'self' already has the data for 0 cell translation
            for _ in 1..sc.raw.periods()[axis] {
                permuted_fcs = permuted_fcs.permuted_by(&deperm);
                self.0 = self.0 + permuted_fcs.0.clone();
            }
        }
        assert_eq!(self.0.row.len(), old_len * sc.raw.num_cells());
        self
    }

    pub fn to_dense_matrix(&self) -> Vec<Vec<M33>> {
        self.0.to_dense()
    }

    pub fn from_dense_matrix(dense: Vec<Vec<M33>>) -> Self {
        SuperForceConstants(RawCoo::from_dense(dense))
    }

    pub fn into_transpose(self) -> Self {
        SuperForceConstants(self.0.into_raw_transpose().map(|m| m.t()))
    }

    /// Drops all of the rows outside `DESIGNATED_CELL`, to recover the standard `ForceConstants`
    /// type on which most methods are implemented.
    ///
    /// **Warning:** A `SuperForceConstants` may have a set of nonzero rows that does not coincide
    /// with the sites in `DESIGNATED_CELL` (e.g. it might use the set of sites displaced by
    /// Phonopy). This method should not be used in such cases, or else data will be lost!.
    pub fn drop_non_designated_rows(self, sc: &SupercellToken) -> ForceConstants {
        let SuperForceConstants(RawCoo { dim: _, row, col, val }) = self;

        let sc = SupercellWrapper::new(sc);
        let cells = sc.atom_cells();
        let primitive_atoms = sc.atom_primitive_atoms();

        let (mut out_row, mut out_col, mut out_val) = (vec![], vec![], vec![]);
        for ((super_r, c), x) in row.into_iter().zip(col).zip(val) {
            if cells[super_r] == DESIGNATED_CELL {
                out_row.push(primitive_atoms[super_r]);
                out_col.push(c);
                out_val.push(x);
            }
        }

        let out_dim = (sc.raw.num_primitive_atoms(), sc.raw.num_supercell_atoms());
        let coo = RawCoo { dim: out_dim, val: out_val, row: out_row, col: out_col };

        ForceConstants(coo.into_csr())
    }
}

// both the rows and columns of ForceConstants are conceptually indexed
// by the same index type, so the Permute impl permutes both.
impl Permute for SuperForceConstants {
    fn permuted_by(self, perm: &Perm) -> SuperForceConstants {
        let SuperForceConstants(RawCoo { dim, mut row, mut col, val }) = self;
        for SuperI(r) in &mut row {
            *r = perm.permute_index(*r);
        }
        for SuperI(c) in &mut col {
            *c = perm.permute_index(*c);
        }
        SuperForceConstants(RawCoo { dim, row, col, val })
    }
}

// ------------------------------------------------------

impl ForceConstants {
    /// Imposes translational invariance like Phonopy.
    ///
    /// **Warning:** This causes the ForceConstants matrix to effectively become dense!
    pub fn impose_translational_invariance(mut self, sc: &SupercellToken, level: u32) -> Self {
        let sc = SupercellWrapper::new(sc);
        for _ in 0..level {
            for _ in 0..2 {
                self.transpose_mut(sc);
                self.cancel_row_averages(sc);
            }
            self.impose_matrix_symmetry(sc);
        }
        self.impose_translational_symmetry(sc);
        self
    }

    // this corresponds to the explicit for loops in py_perm_trans_symmetrize_compact_fc
    // in _phonopy.c: https://github.com/atztogo/phonopy/blob/4fbd156b705c/c/_phonopy.c#L564-L577
    //
    /// Imposes that the sum over each row is the zero matrix, by subtracting the mean for each row.
    /// (i.e. setting the 0th fourier component to zero)
    ///
    /// **Warning:** This causes the ForceConstants matrix to effectively become dense!
    fn cancel_row_averages(&mut self, sc: SupercellWrapper<'_>) {
        let mut bee = self.0.to_coo().into_bee();
        for row in bee.map.values_mut() {
            let mean = row.values().fold(M33::zero(), |acc, m| &acc + m) / sc.raw.num_supercell_atoms() as f64;

            for super_c in sc.super_indices() {
                *row.entry(super_c).or_insert_with(M33::zero) -= mean;
            }
        }
        self.0 = bee.into_csr();
    }

    // this corresponds to set_translational_symmetry_compact_fc in _phonopy.c
    //
    /// If `self` is a symmetric matrix, this imposes that the sum over each row and column is the
    /// zero matrix, by ignoring the existing values in the submatrices along the main diagonal and
    /// replacing them with values computed from the rest of the row and column.
    fn impose_translational_symmetry(&mut self, sc: SupercellWrapper<'_>) {
        let mut bee = self.0.to_coo().into_bee();
        for (&prim_r, row) in bee.map.iter_mut() {
            let super_r = sc.designated_super(prim_r);

            // compute the sum without the diagonal submatrix
            row.insert(super_r, M33::zero());
            let sum = row.values().fold(M33::zero(), |acc, m| &acc + m);

            // (to my understanding, the .t() here represents the sum over columns,
            //  on the precondition that `self` is symmetric)
            row.insert(super_r, -(sum + sum.t()) / 2.0);
        }
        self.0 = bee.into_csr();
    }

    // this corresponds to set_index_permutation_symmetry_compact_fc(is_transpose=1) in _phonopy.c
    //
    /// Transposes the matrix in-place.
    fn transpose_mut(&mut self, sc: SupercellWrapper<'_>,) {
        self.visit_with_transpose(sc, |&m, &t| (t, m))
    }

    // this corresponds to set_index_permutation_symmetry_compact_fc(is_transpose=0) in _phonopy.c
    //
    /// Imposes that `M = M.T` by setting self equal to `(M + M.T) / 2`.
    fn impose_matrix_symmetry(&mut self, sc: SupercellWrapper<'_>,) {
        self.visit_with_transpose(sc, |&m, &t| {
            let average = (m + t) / 2.0;
            (average, average)
        })
    }

    // this corresponds to set_index_permutation_symmetry_compact_fc in _phonopy.c
    //
    /// Finds each pair `(m, t)` of matrices that are supposed to be transposes of each other,
    /// and calls a function to obtain new values for them.
    ///
    /// The input function must have the following properties:
    ///
    /// * If the inputs are reversed, the outputs are reversed.
    ///   (`visit(t, m) == flip(visit(m, t))`, where `flip((x, y)) = (y, x)`)
    /// * If both inputs are transposed, the outputs are transposed.
    /// * It is linear.  This implies `visit(zero, zero) == (zero, zero)`.
    /// * `visit(m, m.t())` must return matrices that are transposes of each other.
    ///
    /// **Warning:** The current implementation causes the ForceConstants matrix to effectively
    /// become dense! (this can be avoided for this specific function, but I haven't bothered
    /// fixing it because there are other functions where it can't be avoided...)
    fn visit_with_transpose(
        &mut self,
        sc: SupercellWrapper<'_>,
        mut visit: impl FnMut(&M33, &M33) -> (M33, M33),
    ) {
        let mut done: Indexed<PrimI, Vec<BTreeSet<SuperI>>>;
        done = Indexed::from_elem_n(BTreeSet::new(), sc.raw.num_primitive_atoms());

        let primitive_atoms = sc.atom_primitive_atoms();
        let lattice_points = sc.atom_lattice_points();

        // Phonopy iterates by column first.  However, due to the symmetry of `visit`,
        // this has no impact on the output. (it only affects whether some matrices are given as
        // the first argument or as the second argument to the function)
        let mut bee = self.0.to_coo().into_bee();
        for prim_r in sc.primitive_indices() {
            let super_r = sc.designated_super(prim_r);
            for super_c in sc.super_indices() {
                if !done[prim_r].contains(&super_c) {
                    // Find the indices of the transposed element.
                    let (prim_r_t, super_c_t) = get_effective_indices(
                        sc, &primitive_atoms, &lattice_points,
                        (super_c, super_r),
                    );
                    assert!(!done[prim_r_t].contains(&super_c_t));

                    // NOTE: These get_muts are what cause this function to make the matrix dense,
                    //       since they are done to all indices.
                    let m = *bee.get_mut(prim_r, super_c);
                    let t = bee.get_mut(prim_r_t, super_c_t).t();
                    let (new_m, new_t) = visit(&m, &t);
                    *bee.get_mut(prim_r, super_c) = new_m;
                    *bee.get_mut(prim_r_t, super_c_t) = new_t.t();

                    done[prim_r].insert(super_c);
                    done[prim_r_t].insert(super_c_t);
                }
            }
        }
        self.0 = bee.into_csr();
    }
}

// Get indices into the stored data for the submatrix that is identical (according to
// translational symmetry) to the element that *would* be located at the given indices.
fn get_effective_indices(
    sc: SupercellWrapper<'_>,
    primitive_atoms: &Indexed<SuperI, [PrimI]>,
    lattice_points: &Indexed<SuperI, [V3<i32>]>,
    (orig_r, orig_c): (SuperI, SuperI),
) -> (PrimI, SuperI) {
    // apply a uniform translation to both indices that maps orig_r into DESIGNATED_CELL
    let correction: V3<i32> = {
        let desired = sc.designated_lattice_point;
        let actual = lattice_points[orig_r];
        desired - actual
    };

    let final_c = sc.atom_from_lattice_point(
        primitive_atoms[orig_c],
        lattice_points[orig_c] + correction,
    );
    (primitive_atoms[orig_r], final_c)
}

// ------------------------------------------------------

impl DynamicalMatrix {
    // max absolute value of M - M.H
    //
    // I saw a value of up to 1e-4 times the maximum matrix element while
    // debugging this; but hermitianizing produced something uniformly within
    // 1e-14 of phonopy's matrix, so I believe this is normal.
    #[allow(unused)]
    pub fn max_hermitian_error(&self) -> f64 {
        let coo_1 = self.0.to_coo();
        let coo_2 = self.conj_t().0.into_coo();
        let difference = coo_1 + coo_2.map(|c| -c);
        difference.val.into_iter()
            .fold(0.0, |mut acc, Complex33(real, imag)| {
                for i in 0..3 {
                    for k in 0..3 {
                        let r2 = real[i][k] * real[i][k];
                        let i2 = imag[i][k] * imag[i][k];
                        acc = f64::max(acc, r2 + i2);
                    }
                }
                acc
            })
            .sqrt()
    }

    pub fn num_atoms(&self) -> usize
    { self.0.dim.0 }

    pub fn hermitianize(&self) -> Self {
        let coo_1 = self.0.to_coo();
        let coo_2 = self.conj_t().0.into_coo();
        let csr = (coo_1 + coo_2).into_csr().map(|mut c| {
            c.0 *= 0.5;
            c.1 *= 0.5;
            c
        });
        DynamicalMatrix(csr)
    }

    pub fn conj_t(&self) -> Self {
        let csr = self.0.to_raw_transpose().map(|c| c.conj_t());
        DynamicalMatrix(csr)
    }

    pub fn cereal(&self) -> Cereal {
        self.0.validate().expect("(BUG!) invalid sparse data");
        Cereal {
            dim: self.0.dim,
            complex_blocks: self.0.val.to_vec(),
            col: index_cast(self.0.col.to_vec()),
            row_ptr: self.0.row_ptr.raw.to_vec(),
        }
    }

    pub fn from_cereal(cereal: Cereal) -> FailResult<Self> {
        let csr = RawCsr {
            dim: cereal.dim,
            val: cereal.complex_blocks,
            col: index_cast(cereal.col),
            row_ptr: Indexed::from_raw(cereal.row_ptr),
        };
        csr.validate()?;

        Ok(DynamicalMatrix(csr))
    }

    /// If the matrix is real, produce a flat `Vec` representation.
    pub fn to_dense_flat_real(&self) -> Option<Vec<f64>> {
        let DynamicalMatrix(RawCsr { dim, val, col, row_ptr }) = self;

        let mut out = vec![0.0; 3 * dim.0 * 3 * dim.1];
        let row_los = &row_ptr.raw[..row_ptr.len() - 1];
        let row_his = &row_ptr.raw[1..];
        for (block_row, (&lo, &hi)) in zip_eq!(row_los, row_his).enumerate() {
            for (block, &PrimI(block_col)) in zip_eq!(&val[lo..hi], &col[lo..hi]) {
                let Complex33(real, imag) = block;
                if imag != &M33::zero() {
                    return None;
                }
                for r in 0..3 {
                    for c in 0..3 {
                        let out_r = 3 * block_row + r;
                        let out_c = 3 * block_col + c;
                        out[out_r * 3 * dim.0 + out_c] = real[r][c];
                    }
                }
            }
        }

        Some(out)
    }

    pub fn compute_eigensolutions_dense_gamma(&self) -> (Eigenvalues, Vec<Vec<V3>>) {
        trace!("Computing all eigensolutions.");
        let mut flat = self.to_dense_flat_real().expect("(BUG!) expected real matrix!");
        let mut eigenvalues = vec![f64::NAN; 3 * self.num_atoms()];
        let mut eigenvectors_flat = vec![f64::NAN; flat.len()];

        rsp2_linalg::dynmat::diagonalize_real(&mut flat, &mut eigenvalues, &mut eigenvectors_flat);

        // save that precious memory!
        drop(flat);

        let eigenvectors = eigenvectors_flat.chunks(3 * self.num_atoms())
            .map(|data| data.nest().to_vec()).collect();

        (Eigenvalues { eigenvalues }, eigenvectors)
    }
}

/// Trivial wrapper type to help ensure eigenvalues don't get mistaken for frequencies.
pub struct Eigenvalues { pub eigenvalues: Vec<f64> }

// ------------------------------------------------------

/// Wraps SupercellToken with methods that use newtype indices
#[derive(Copy, Clone)]
struct SupercellWrapper<'a> {
    raw: &'a SupercellToken,
    designated_lattice_point: V3<i32>,
}

impl<'a> SupercellWrapper<'a> {
    fn new(raw: &'a SupercellToken) -> Self {
        let designated_lattice_point = raw.lattice_point_from_cell(DESIGNATED_CELL);
        Self { raw, designated_lattice_point }
    }

    fn primitive_indices(&self) -> impl Iterator<Item=PrimI> {
        (0..self.raw.num_primitive_atoms()).map(PrimI)
    }

    fn super_indices(&self) -> impl Iterator<Item=SuperI> {
        (0..self.raw.num_supercell_atoms()).map(SuperI)
    }

    // (note: lattice_point is wrapped into the supercell)
    fn atom_from_lattice_point(&self, prim: PrimI, lattice_point: V3<i32>) -> SuperI {
        SuperI::new(self.raw.atom_from_lattice_point(prim.index(), lattice_point))
    }

    fn designated_super(&self, prim: PrimI) -> SuperI {
        self.atom_from_lattice_point(prim, self.designated_lattice_point)
    }

    fn atom_cells(&self) -> Indexed<SuperI, Vec<[u32; 3]>> {
        Indexed::from_raw(self.raw.atom_cells())
    }

    fn atom_primitive_atoms(&self) -> Indexed<SuperI, Vec<PrimI>> {
        Indexed::from_raw(index_cast(self.raw.atom_primitive_atoms()))
    }

    fn atom_lattice_points(&self) -> Indexed<SuperI, Vec<V3<i32>>> {
        Indexed::from_raw(self.raw.atom_lattice_points())
    }
}

// ------------------------------------------------------

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Cereal {
    // these should be suitable for col and row_ptr (i.e. no additional factor of 3).
    pub dim: (usize, usize),

    // CSR format (or technically Block CSR).
    pub complex_blocks: Vec<Complex33>,
    pub col: Vec<usize>,
    pub row_ptr: Vec<usize>,
}

// FIXME: Now that we've decided to add num_complex as a dependency,
//        this type is kinda weird to have around.
pub use self::complex_33::Complex33;
mod complex_33 {
    use super::*;

    // element type of the dynamical matrix, used to shoehorn it into a sparse matrix container
    #[derive(Debug, Copy, Clone, PartialEq)]
    #[derive(Serialize, Deserialize)]
    pub struct Complex33(pub M33, pub M33);

    impl num_traits::Zero for Complex33 {
        fn zero() -> Self { Complex33(M33::zero(), M33::zero()) }
        fn is_zero(&self) -> bool { self.0.is_zero() && self.1.is_zero() }
    }

    impl Complex33 {
        pub fn conj_t(&self) -> Self
        { Complex33(self.0.t(), -self.1.t()) }
    }

    impl std::ops::Add for Complex33 {
        type Output = Complex33;

        fn add(mut self, rhs: Complex33) -> Self::Output
        { self += rhs; self }
    }

    impl std::ops::Sub for Complex33 {
        type Output = Complex33;

        fn sub(self, rhs: Complex33) -> Self::Output
        { self + -rhs }
    }

    impl std::ops::Neg for Complex33 {
        type Output = Complex33;

        fn neg(self: Complex33) -> Self::Output
        { Complex33(-self.0, -self.1) }
    }

    impl std::ops::AddAssign for Complex33 {
        fn add_assign(&mut self, Complex33(real, imag): Complex33) {
            self.0 += real;
            self.1 += imag;
        }
    }
}

// ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rsp2_structure::{supercell, Lattice, CoordsKind};

    fn make_fc_test_data() -> (ForceConstants, SupercellToken) {
        // big enough that permutations of translation don't equal their inverse
        let sc_dim = [3, 3, 1];

        let prim_coords = Coords::new(Lattice::eye(), CoordsKind::Carts(vec![V3::zero(); 2]));
        let sc = supercell::diagonal(sc_dim).build(&prim_coords).1;

        // generate random force constants
        let mut rng = rand::thread_rng();
        let map = (0..sc.num_primitive_atoms()).map(|prim_r| {
            let row = (0..sc.num_supercell_atoms()).filter_map(|super_c| {
                if rng.next_f64() > 0.3 {
                    Some((SuperI(super_c), M33::from_fn(|_, _| 2.0 * rng.next_f64() - 1.0)))
                } else {
                    None
                }
            }).collect();
            (PrimI(prim_r), row)
        }).collect();

        let dim = (sc.num_primitive_atoms(), sc.num_supercell_atoms());
        let bee = RawBee { map, dim };

        (ForceConstants(bee.to_csr()), sc)
    }

    #[test]
    fn fc_transpose() {
        let (orig, sc) = make_fc_test_data();

        // take the transpose first (tricky), then add the missing rows
        let actual = {
            let mut modified = orig.clone();
            modified.transpose_mut(SupercellWrapper::new(&sc));
            modified.to_super_force_constants_with_all_rows(&sc)
        };

        // add the missing rows, then take the transpose (simple)
        let expected = {
            let super_fcs = orig.to_super_force_constants_with_all_rows(&sc);
            super_fcs.into_transpose()
        };

        assert_eq!(expected.to_dense_matrix(), actual.to_dense_matrix());
    }

    #[test]
    fn fc_impose_matrix_symmetry() {
        let (orig, sc) = make_fc_test_data();

        // symmetrize first (tricky), then add the missing rows
        let actual = {
            let mut modified = orig.clone();
            modified.impose_matrix_symmetry(SupercellWrapper::new(&sc));
            modified.to_super_force_constants_with_all_rows(&sc)
        };

        // add the missing rows, then symmetrize (simple)
        let expected = {
            let super_fcs = orig.to_super_force_constants_with_all_rows(&sc);
            SuperForceConstants({
                (super_fcs.0.clone() + super_fcs.into_transpose().0)
                    .map(|m| m / 2.0)
            })
        };

        assert_eq!(expected.to_dense_matrix(), actual.to_dense_matrix());
    }
}
