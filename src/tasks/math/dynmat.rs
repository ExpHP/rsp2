
// TODO:
// Required bits of cleanup here:
//
// * breakitup

use ::FailResult;
use ::meta::Mass;
use ::rsp2_array_types::{V3, M33, M3};
use ::rsp2_soa_ops::{Perm, Permute};
use ::rsp2_structure::supercell::SupercellToken;
use ::rsp2_newtype_indices::{Idx, Indexed, cast_index};
use ::std::rc::Rc;
use ::std::collections::BTreeMap;
use ::slice_of_array::prelude::*;
use ::math::sparse::{self, RawCoo, RawCsr};

use ::util::ext_traits::OptionExpectNoneExt;

// Displaced atoms are always in this cell.
// (HACK: shouldn't need to be pub(crate))
pub(crate) const DESIGNATED_CELL: [u32; 3] = [0, 0, 0];

// index types for local use, to aid in reasoning
newtype_index!{DispI}  // index of a displacement
newtype_index!{PrimI}  // index of a primitive cell site
newtype_index!{SuperI} // index of a supercell site
newtype_index!{StarI}  // index of a star of sites equivalent under symmetry
newtype_index!{OperI}  // index into the space group
newtype_index!{EqnI}   // row of a table of equations that will be solved by pseudo inverse

#[derive(Debug, Clone)]
pub struct ForceConstants(
    RawCoo<M33, SuperI, SuperI>,
);

#[derive(Debug, Clone)]
pub struct DynamicalMatrix(
    pub RawCsr<Complex33, PrimI, PrimI>,
);

impl ForceConstants {
    // This basically follows Phonopy's method for `--writefc` to a T, although it
    // is a bit smarter in some places, and probably dumber in other places.
    //
    // Notable differences from phonopy include:
    //
    // * This uses a sparse format for anything indexed by atoms (primitive or super)
    // * Phonopy translates each displaced atom to the origin and performs
    //   expensive brute force overlap searches, while this leaves everything where
    //   they are and adjusts indices to simulate translation by a lattice point)
    //
    // The produced ForceConstants only include the rows in the 'designated cell.'
    // (the only rows that are required to produce a dynamical matrix)
    pub(crate) fn like_phonopy(
        // Displacements, using primitive cell indices.
        //
        // For each symmetry star of sites in the primitive cell, only one of those
        // sites may appear in this list. (this requirement is checked)
        prim_displacements: &[(usize, V3)], // [displacement] -> (prim_displaced, cart_disp)
        force_sets: &[BTreeMap<usize, V3>], // [displacement][affected] -> cart_force
        cart_rots: &[M33],                  // [sg_index] -> matrix
        super_deperms: &[Perm],             // [sg_index] -> perm
        sc: &SupercellToken,
    ) -> FailResult<ForceConstants>
    {
        // wrap data with information about index type
        //
        // most type annotations in here are not strictly necessary, but serve as a stop-gap measure
        // to ensure that at least *something* close to the public interface stops compiling if the
        // newtyped indices in Context are changed (to remind you to check callers)
        let displacements: &[(PrimI, V3)] = cast_index(prim_displacements);
        let displacements: &Indexed<DispI, [_]> = Indexed::from_raw_ref(displacements);

        let force_sets: &[BTreeMap<SuperI, V3>] = cast_index(force_sets);
        let force_sets: &Indexed<DispI, [_]> = Indexed::from_raw_ref(force_sets);

        let cart_rots:     &Indexed<OperI, [M33]> = Indexed::from_raw_ref(cart_rots);
        let super_deperms: &Indexed<OperI, [Perm]> = Indexed::from_raw_ref(super_deperms);

        let primitive_atoms: Indexed<SuperI, Vec<PrimI>>;
        let lattice_points:  Indexed<SuperI, Vec<V3<i32>>>;
        primitive_atoms = Indexed::from_raw(cast_index(sc.atom_primitive_atoms()));
        lattice_points = Indexed::from_raw(sc.atom_lattice_points());
        let primitive_atoms = &primitive_atoms[..];
        let lattice_points = &lattice_points[..];

        let designated_lattice_point = sc.lattice_point_from_cell(DESIGNATED_CELL);

        Context {
            sc, primitive_atoms, lattice_points, displacements, force_sets,
            cart_rots, super_deperms,
            designated_lattice_point,
        }.like_phonopy()
    }
}

// a variety of precomputed things made available during the whole computation
struct Context<'ctx> {
    // Function arguments
    displacements:   &'ctx Indexed<DispI, [(PrimI, V3)]>,
    force_sets:      &'ctx Indexed<DispI, [BTreeMap<SuperI, V3>]>,
    cart_rots:       &'ctx Indexed<OperI, [M33]>,
    super_deperms:   &'ctx Indexed<OperI, [Perm]>,
    sc:              &'ctx SupercellToken,

    // Some data from `sc` wrapped with newtyped indices
    primitive_atoms: &'ctx Indexed<SuperI, [PrimI]>,
    lattice_points:  &'ctx Indexed<SuperI, [V3<i32>]>,

    // We only bother populating the rows of the force constants matrix
    // that correspond to displaced atoms in a single image of the primitive cell.
    designated_lattice_point: V3<i32>,
}

impl<'ctx> Context<'ctx> {
    // FIXME break up
    fn like_phonopy(
        &self,
    ) -> FailResult<ForceConstants> {

        #[derive(Debug, Clone)]
        struct StarData {
            representative: PrimI,
            // operators that map the representative atom into this position.
            displacements: Vec<DispI>,
        }

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

        // Find relationships between each primitive site and their site-symmetry representatives.
        #[derive(Debug, Clone)]
        struct PrimData {
            star: StarI,
            // operators that move the star's representative into this primitive site.
            opers_from_rep: Vec<OperI>,
        }

        let prim_data: Indexed<PrimI, Vec<PrimData>> = {
            let mut data = Indexed::<PrimI, _>::from_elem_n(None, self.sc.num_primitive_atoms());
            for (star, star_data) in star_data.iter_enumerated() {
                let representative_atom = self.atom_from_lattice_point(star_data.representative, self.designated_lattice_point);
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

        let mut computed_rows: BTreeMap<PrimI, BTreeMap<SuperI, M33>> = Default::default();

        // Populate the rows belonging to symmetry star representatives.
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

            computed_rows.insert(representative, force_constants_row)
                .expect_none("(BUG) computed same row of FCs twice!?");
        } // for star

        for (prim, data) in prim_data.into_iter_enumerated() {
            let PrimData { star, opers_from_rep } = data;
            match computed_rows.contains_key(&prim) {
                true => continue,
                false => {
                    let representative = star_data[star].representative;
                    let &oper = opers_from_rep.get(0).expect("(BUG!) was checked earlier");
                    let new_row = self.derive_row_by_symmetry(
                        representative,
                        &computed_rows[&representative],
                        prim,
                        oper,
                    );

                    computed_rows.insert(prim, new_row)
                        .expect_none("(BUG) computed same row of FCs twice!?");
                },
            };
        }

        assert!(computed_rows.keys().cloned().eq(self.prim_indices()));
        let matrix = {
            let dim = (self.sc.num_supercell_atoms(), self.sc.num_supercell_atoms());
            let map = {
                computed_rows.into_iter()
                    .map(|(prim, row_map)| (self.designated_super(prim), row_map))
                    .collect()
            };
            sparse::RawBee { dim, map }.into_coo()
        };
        Ok(ForceConstants(matrix))
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
        // done mutating
        let all_displacements = all_displacements;
        let all_sparse_forces = all_sparse_forces;

        // now that all equations have been built, we can "densify" the force matrices.
        // (with respect to equations; not the entire supercell, of course!)
        let num_eqns = all_displacements.len();
        let all_forces = {
            all_sparse_forces.into_iter()
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
                }).collect()
        };
        (all_displacements, all_forces)
    }

    fn derive_row_by_symmetry(
        &self,
        // Index and data of a finished row in the force constants table
        finished_prim: PrimI,
        finished_row: &BTreeMap<SuperI, M33>,
        // The row to be computed. (for sanity checks)
        todo_prim: PrimI,
        // An arbitrarily chosen operator that maps `finished_prim` to `todo_prim`
        // in the primitive structure.
        oper: OperI,
    ) -> BTreeMap<SuperI, M33>
    {
        // Visualize the operation described in the documentation of `get_corrected_rotate`.
        //
        // Performing this operation on a set of data would permute all indices according to the
        // deperm returned by `get_corrected_rotate`, and would rotate all displacements and forces
        // by the rotation part of the operator.  How does this impact the force constants?
        //
        // Clearly, the columns will be permuted by the deperm:
        let (rotated_prim, apply_deperm) = self.get_corrected_rotate(oper, finished_prim);
        assert_eq!(rotated_prim, todo_prim, "(BUG) oper produced different row?!");

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

        finished_row.iter()
            .map(|(&affected, fc_matrix)| {
                let affected = apply_deperm(affected);
                let fc_matrix = cart_rot * fc_matrix * cart_rot.t();
                (affected, fc_matrix)
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
        let displaced_super = self.atom_from_lattice_point(displaced, self.designated_lattice_point);

        let correction: V3<i32> = {
            let desired = self.designated_lattice_point;
            let actual = self.lattice_points[self.rotate_atom(oper, displaced_super)];
            desired - actual
        };

        let rotate_and_translate = move |atom| {
            let atom = self.rotate_atom(oper, atom);
            self.atom_from_lattice_point(
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
    fn prim_indices(&self) -> impl Iterator<Item=PrimI>
    { (0..self.sc.num_primitive_atoms()).map(PrimI::new) }

    // depermutations in the form of a function that maps sparse indices
    fn rotate_atom(&self, oper: OperI, atom: SuperI) -> SuperI {
        let deperm = &self.super_deperms[oper];
        SuperI::new(deperm.permute_index(atom.index()))
    }

    // (note: lattice_point is wrapped into the supercell)
    fn atom_from_lattice_point(&self, prim: PrimI, lattice_point: V3<i32>) -> SuperI {
        SuperI::new(self.sc.atom_from_lattice_point(prim.index(), lattice_point))
    }

    fn designated_super(&self, prim: PrimI) -> SuperI {
        self.atom_from_lattice_point(prim, self.designated_lattice_point)
    }

    // recovers the primitive index from a supercell index whose lattice point is
    // `designated_lattice_point`. (panics if this does not hold, in which case you
    // may have forgotten to apply a translation or something)
    fn prim_from_designated_super(&self, atom: SuperI) -> PrimI {
        assert_eq!(self.designated_lattice_point, self.lattice_points[atom]);
        self.primitive_atoms[atom]
    }
}

#[allow(unused)]
impl ForceConstants {
    // Take ForceConstants where row_atom is always in DISPLACED_CELL
    // and generate all the other rows.
    //
    // NOTE: This is just here for the lulz, in case you need something easier to
    //       compare against phonopy pre-12.8.  The rows outside of the designated
    //       cell are merely permuted forms of the designated rows, and do not even
    //       appear in the formula for the dynamical matrix.
    //
    //       Adding them *could* make cleaning the FCs easier, if we wanted to simulate
    //       phonopy's FC_SYMMETRY, though I think that *even then* they are still
    //       unnecessary. Phonopy's `perm_trans_symmetrize_fc` C function does something
    //       clever (and obscure, and undocumented...) to impose translational invariance
    //       on both the rows and columns by only using the data within a row.
    //       (together with the fact that it just symmetrized the matrix)
    #[allow(unused)]
    pub fn add_rows_for_other_cells(mut self, sc: &SupercellToken) -> Self {
        assert!({
            let cells = sc.atom_cells();
            self.0.row.iter().all(|&SuperI(row)| cells[row] == DESIGNATED_CELL)
        });

        let old_len = self.0.row.len();
        for axis in 0..3 {
            // get deperm that translates data by one cell along this axis
            let unit = V3::from_fn(|i| (i == axis) as i32);
            let deperm = sc.lattice_point_translation_deperm(unit);

            let mut permuted_fcs = self.clone();

            // skip 0 because 'self' already has the data for 0 cell translation
            for _ in 1..sc.periods()[axis] {
                permuted_fcs = permuted_fcs.permuted_by(&deperm);
                self.0 = self.0 + permuted_fcs.0.clone();
            }
        }
        assert_eq!(self.0.row.len(), old_len * sc.num_cells());
        self
    }

    // HACK
    fn from_dense_matrix(mat: Vec<Vec<M33>>) -> ForceConstants {
        let nrows = mat.len();
        let ncols = mat.get(0).expect("cant sparsify matrix with no rows").len();
        let dim = (nrows, ncols);

        let (mut row, mut col, mut val) = (vec![], vec![], vec![]);
        for (r, row_vec) in mat.into_iter().enumerate() {
            assert_eq!(row_vec.len(), ncols);
            for (c, m) in row_vec.into_iter().enumerate() {
                if m != M33::zero() {
                    row.push(SuperI(r));
                    col.push(SuperI(c));
                    val.push(m);
                }
            }
        }
        ForceConstants(RawCoo { dim, row, col, val })
    }

    // note: other K points will require cartesian coords for the right phase factors
    /// Compute the dynamical matrix at gamma.
    ///
    /// The force constants do not need to contain data for rows outside the
    /// designated cell. (but if they do, it won't hurt)
    pub fn gamma_dynmat(
        &self,
        sc: &SupercellToken,
        masses: Rc<[Mass]>,
    ) -> DynamicalMatrix {
        assert_eq!(masses.len(), sc.num_primitive_atoms());
        let masses: &Indexed<PrimI, _> = Indexed::from_raw_ref(&masses[..]);

        let sc = sc.clone();

        let primitive_atoms = sc.atom_primitive_atoms();
        let cells = sc.atom_cells();
        let get_prim = |SuperI(r)| PrimI(primitive_atoms[r.index()]);

        let iter = zip_eq!(&self.0.row, &self.0.col, &self.0.val)
            // ignore elements outside the rows of the designated cell, which were added
            // for no other purpose than to facilitate imposing translational invariance
            .filter(|&(&SuperI(r), _, _)| cells[r] == DESIGNATED_CELL)
            // each column of the dynamical matrix sums over columns for images in
            // the force constants matrix, with phase factors.
            .map(|(&r, &c, &m)| {
                let r = get_prim(r);
                let c = get_prim(c);

                // mass-normalizing scale factor
                let scale = 1.0 / f64::sqrt(masses[r].0 * masses[c].0);

                // at gamma, phase is 1
                let (phase_real, phase_imag) = (1.0, 0.0);
                let real = scale * phase_real * m;
                let imag = scale * phase_imag * m;

                ((r, c), Complex33(real, imag))
            });

        let matrix = {
            let (pos, val): (Vec<_>, Vec<_>) = iter.unzip();
            let (row, col) = pos.into_iter().unzip();
            let dim = (sc.num_primitive_atoms(), sc.num_primitive_atoms());
            RawCoo { dim, val, row, col }.into_csr()
        };

        DynamicalMatrix(matrix)
    }
}

// both the rows and columns of ForceConstants are conceptually indexed
// by the same index type, so the Permute impl permutes both.
impl Permute for ForceConstants {
    fn permuted_by(self, perm: &Perm) -> ForceConstants {
        let ForceConstants(RawCoo { dim, mut row, mut col, val }) = self;
        for SuperI(r) in &mut row {
            *r = perm.permute_index(*r);
        }
        for SuperI(c) in &mut col {
            *c = perm.permute_index(*c);
        }
        ForceConstants(RawCoo { dim, row, col, val })
    }
}

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
}

pub use self::complex_33::Complex33;
mod complex_33 {
    use super::*;

    // element type of the dynamical matrix, used to shoehorn it into a sparse matrix container
    #[derive(Debug, Copy, Clone)]
    #[derive(Serialize, Deserialize)]
    pub struct Complex33(pub M33, pub M33);

    impl ::num_traits::Zero for Complex33 {
        fn zero() -> Self { Complex33(M33::zero(), M33::zero()) }
        fn is_zero(&self) -> bool { self.0.is_zero() && self.1.is_zero() }
    }

    impl Complex33 {
        pub fn conj_t(&self) -> Self
        { Complex33(self.0.t(), -self.1.t()) }
    }

    impl ::std::ops::Add for Complex33 {
        type Output = Complex33;

        fn add(mut self, rhs: Complex33) -> Self::Output
        { self += rhs; self }
    }

    impl ::std::ops::Sub for Complex33 {
        type Output = Complex33;

        fn sub(self, rhs: Complex33) -> Self::Output
        { self + -rhs }
    }

    impl ::std::ops::Neg for Complex33 {
        type Output = Complex33;

        fn neg(self: Complex33) -> Self::Output
        { Complex33(-self.0, -self.1) }
    }

    impl ::std::ops::AddAssign for Complex33 {
        fn add_assign(&mut self, Complex33(real, imag): Complex33) {
            self.0 += real;
            self.1 += imag;
        }
    }
}
