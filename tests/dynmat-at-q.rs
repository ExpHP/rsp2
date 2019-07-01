use ::rsp2_integration_test::{CliTest, filetypes, resource, cli_test, Result};

// Check rsp2 against dynmats produced by phonopy.
//
// The workflow to update the test outputs is a bit rough:
//
//    * phonopy -d --dim "13 13 1" --amplitude 1e-2
//    * Make band.conf:
//          EIGENVECTORS = .TRUE.
//          DIM = 13 13 1
//          BAND = 0 0 0   1/3 1/3 0   1/2 0 0  -1/2 0 0  0 0 0
//          BAND_POINTS = 1
//    * Phonopy's FORCE_SETS are written to low precision, so get one at high precision by
//      setting `RUST_LOG=rsp2_tasks::special::phonopy_force_sets=trace` while running this test.
//    * Hack phonopy.phonon.band_structure.BandStructure._solve_dm_on_path to write each
//      dm to .npz files.  This must be done using scipy.sparse.save_npz on a BSR matrix
//      created with blocksize=(3,3).
//    * phonopy band.conf

#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn dynmat_at_q() -> Result<()> {
    let env = cli_test::Environment::init();

    for &(ref expected_outfile, kpoint) in &[
        (resource("dynmat-at-q/dynmat-gamma.npz"), "0 0 0"),
        (resource("dynmat-at-q/dynmat-k.npz"), "1/3 1/3 0"),
        (resource("dynmat-at-q/dynmat-m.npz"), "0.5 0 0"),
        (resource("dynmat-at-q/dynmat-m-neg.npz"), "-0.5 0 0"),
    ] {
        println!("Testing kpoint {}", kpoint);
        CliTest::cargo_binary(&env, "rsp2-dynmat-at-q")
            .arg("-c").arg(resource("dynmat-at-q/settings.yaml"))
            .arg(resource("dynmat-at-q/input.structure"))
            .arg("--qpoint").arg(kpoint)
            .arg("-o").arg("dynmat.npz")
            .check_file::<filetypes::Dynmat>(
                "dynmat.npz".as_ref(),
                expected_outfile.as_ref(),
                filetypes::DynmatTolerances {
                    rel_tol: 1e-6,
                    abs_tol: 1e-9,
                },
            )
            .run()?;
    }
    Ok(())
}
