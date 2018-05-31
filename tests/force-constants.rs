extern crate rsp2_array_types;
extern crate rsp2_tasks;
extern crate rsp2_structure;
extern crate rsp2_soa_ops;
#[macro_use]
extern crate rsp2_assert_close;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate failure;

type FailResult<T> = Result<T, ::failure::Error>;
use ::std::path::Path;
use ::std::fs::File;

use ::rsp2_array_types::{M33, V3, Unvee};
use ::rsp2_structure::{Coords, FracOp};
use ::rsp2_soa_ops::{Permute};

use ::rsp2_tasks::exposed_for_testing::ForceConstants;
use ::rsp2_tasks::exposed_for_testing::meta::Mass;

// data affected by choice of primitive structure
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct PrimInfo {
    #[serde(rename =      "frac-ops")] frac_ops: Vec<FracOp>,
    #[serde(rename =     "cart-rots")] cart_rots: Vec<M33>,
    #[serde(rename =        "masses")] prim_masses: Vec<f64>,
    #[serde(rename =     "structure")] prim_coords: Coords,
    #[serde(rename = "displacements")] prim_displacements: Vec<(usize, V3)>, // [disp] -> (prim, v)
}

// data affected by choice of supercell
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct SuperInfo {
    #[serde(rename =         "sc-dims")] sc_dims: [u32; 3],
    // supercell coordinates and "designated cell indices" are used to make the test
    // robust to changes in supercell ordering convention
    #[serde(rename = "designated-cell")] orig_designated_images: Vec<usize>, // [prim] -> super
    #[serde(rename =       "structure")] orig_super_coords: Coords,
    #[serde(rename =      "force-sets")] orig_super_force_sets: Vec<Vec<(usize, V3)>>, // [disp] -> [(super, V3)]
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct OutputForceConstants {
    #[serde(rename = "dense")] orig_force_constants: Vec<Vec<M33>>, // [super][super]
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct OutputDynMat {
    #[serde(rename = "real")] expected_dynmat_real: Vec<Vec<M33>>, // [super][super]
    #[serde(rename = "imag")] expected_dynmat_imag: Vec<Vec<M33>>, // [super][super]
}

fn check(
    prim_info: impl AsRef<Path>,
    super_info: impl AsRef<Path>,
    expected_fc: impl AsRef<Path>,
    expected_dynmat: impl AsRef<Path>,
    rel_tol: f64,
    abs_tol: f64,
) -> FailResult<()> {
    let PrimInfo {
        cart_rots, frac_ops, prim_masses, prim_coords, prim_displacements,
    } = ::serde_json::from_reader(File::open(prim_info)?)?;

    let SuperInfo {
        sc_dims, orig_super_coords, orig_super_force_sets, orig_designated_images,
    } = ::serde_json::from_reader(File::open(super_info)?)?;

    let OutputForceConstants {
        orig_force_constants,
    } = ::serde_json::from_reader(File::open(expected_fc)?)?;

    let OutputDynMat {
        expected_dynmat_real,
        expected_dynmat_imag,
    } = ::serde_json::from_reader(File::open(expected_dynmat)?)?;

    let (super_coords, sc) = ::rsp2_structure::supercell::diagonal(sc_dims).build(&prim_coords);

    // permute expected output into correct order for the current supercell convention
    let perm_from_orig = orig_super_coords.perm_to_match(&super_coords, 1e-10)?;
    let super_force_sets: Vec<_> = {
        orig_super_force_sets
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(atom, v)| (perm_from_orig.permute_index(atom), v))
                    .collect()
            })
            .collect()
    };
    // (permute both the rows and cols of a dense matrix from the original supercell
    // order into the current)
    let permute_block = |m: Vec<Vec<M33>>| {
        m.permuted_by(&perm_from_orig)
            .into_iter().map(|x| x.permuted_by(&perm_from_orig))
            .collect::<Vec<_>>()
    };
    let expected_force_constants: Vec<_> = permute_block(orig_force_constants);

    let super_displacements: Vec<_> = {
        prim_displacements.into_iter()
            .map(|(prim, v3)| {
                let orig_atom = orig_designated_images[prim];
                let atom = perm_from_orig.permute_index(orig_atom);
                (atom, v3)
            })
            .collect()
    };

    let super_sg_deperms: Vec<_> = {
        ::rsp2_structure::find_perm::of_spacegroup_for_general(
            &super_coords,
            &frac_ops,
            &prim_coords.lattice(),
            1e-1,
        )?.into_iter().map(|p| p.inverted()).collect()
    };

    // ---------------------------------
    // ------- Force constants ---------
    // ---------------------------------

    let force_constants = ForceConstants::compute_required_rows(
        &super_displacements,
        &super_force_sets,
        &cart_rots,
        &super_sg_deperms,
        &sc,
    )?;

    {
        let raw = force_constants.to_dense_matrix();
        assert_eq!(raw.len(), sc.num_supercell_atoms());
        assert_eq!(raw[0].len(), sc.num_supercell_atoms());
        for r in 0..sc.num_supercell_atoms() {
            for c in 0..sc.num_supercell_atoms() {
                assert_close!(
                    rel=rel_tol, abs=abs_tol,
                    raw[r][c].unvee(),
                    expected_force_constants[r][c].unvee(),
                    "{:?}", force_constants,
                );
            }
        }
    }

    // ----------------------------------
    // ------- Dynamical Matrix ---------
    // ----------------------------------

    let prim_masses: Vec<_> = prim_masses.into_iter().map(Mass).collect();
    let dynamical_matrix = force_constants.gamma_dynmat(&sc, prim_masses.into());

    {
        let real = dynamical_matrix.0.to_coo().map(|c| c.0).into_dense();
        let imag = dynamical_matrix.0.to_coo().map(|c| c.1).into_dense();
        assert_eq!(real.len(), sc.num_primitive_atoms());
        assert_eq!(real[0].len(), sc.num_primitive_atoms());
        for r in 0..sc.num_primitive_atoms() {
            for c in 0..sc.num_primitive_atoms() {
                assert_close!(
                    rel=rel_tol, abs=abs_tol,
                    real[r][c].unvee(),
                    expected_dynmat_real[r][c].unvee(),
                    "{:?}", (real, imag),
                );
                assert_close!(
                    rel=rel_tol, abs=abs_tol,
                    imag[r][c].unvee(),
                    expected_dynmat_imag[r][c].unvee(),
                    "{:?}", (real, imag),
                );
            }
        }
    }
    Ok(())
}

#[test]
fn graphene_denseforce_771() {
    check(
        // * Graphene
        "tests/resources/force-constants/graphene.primitive.json",
        // * Dense force sets
        // * [7, 7, 1] supercell
        "tests/resources/force-constants/graphene-771-dense.super.json",
        "tests/resources/force-constants/graphene-771.fc.json",
        "tests/resources/force-constants/graphene-gamma.dynmat.json",
        1e-10,
        1e-12,
    ).unwrap()
}

#[test]
fn graphene_denseforce_111() {
    check(
        // * Graphene
        "tests/resources/force-constants/graphene.primitive.json",
        // * Dense force sets
        // * [1, 1, 1] supercell
        "tests/resources/force-constants/graphene-111-dense.super.json",
        "tests/resources/force-constants/graphene-111.fc.json",
        // * same dynmat output as for 771; a supercell should not be necessary
        //   for the dynamical matrix at gamma
        "tests/resources/force-constants/graphene-gamma.dynmat.json",
        1e-10,
        1e-12,
    ).unwrap()
}

#[test]
fn graphene_sparseforce_771() {
    check(
        // * Graphene
        "tests/resources/force-constants/graphene.primitive.json",
        // * Sparse force sets (clipped to zero at 1e-12, ~70% sparse)
        // * [7, 7, 1] supercell
        "tests/resources/force-constants/graphene-771-sparse.super.json",
        "tests/resources/force-constants/graphene-771.fc.json",

        // I don't have data computed specifically for this, just use
        // the existing data with a more relaxed tolerance
        "tests/resources/force-constants/graphene-gamma.dynmat.json",
        1e-6,
        1e-7,
    ).unwrap()
}
