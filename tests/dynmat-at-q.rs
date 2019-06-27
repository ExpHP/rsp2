use ::rsp2_integration_test::{CliTest, filetypes, resource, cli_test, Result};

// Check rsp2 against dynmats produced by phonopy.
//
// The workflow to update the test outputs is a bit rough:
//
//    * Acquire the POSCAR and put in a directory.
//    * phonopy -d --dim "14 14 1"
//    * for n in 001 002 003 004; do
//         cargo run --bin rsp2-compute-for-phonopy -- -c ../settings.yaml POSCAR-$n -o vasprun-${n}.xml
//      done
//    * phonopy band.conf -f vasprun-*.xml
//    * Make band.conf:
//          EIGENVECTORS = .TRUE.
//          DIM = 14 14 1
//          BAND = 0 0 0   1/3 1/3 0   1/2 0 0   0 0 0
//          BAND_POINTS = 1
//    * Hack phonopy.phonon.band_structure.BandStructure._solve_dm_on_path to write each
//      dm to .npz files.  This must be done using scipy.sparse.save_npz on a BSR matrix
//      created with blocksize=(3,3).
//    * phonopy band.conf

#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn dynmat_at_q() -> Result<()> {
    let env = cli_test::Environment::init();

    for &(ref expected_outfile, kpoint) in &[
        (resource("dynmat-at-q/dynmat-gamma.npz"), "[0, 0, 0]"),
        (resource("dynmat-at-q/dynmat-k.npz"), "[0.3333333333333, 0.3333333333333, 0]"),
        (resource("dynmat-at-q/dynmat-m.npz"), "[0.5, 0, 0]"),
    ] {
        println!("Testing kpoint {}", kpoint);
        CliTest::cargo_binary(&env, "rsp2")
            .arg("-c").arg(resource("dynmat-at-q/settings.yaml"))
            .arg(resource("dynmat-at-q/initial.structure"))
            .arg("--qpoint").arg(kpoint)
            .arg("-o").arg("dynmat.npz")
            .check_file::<filetypes::Dynmat>(
                "out/gammat.npz".as_ref(),
                expected_outfile.as_ref(),
                filetypes::DynmatTolerances {
                    rel_tol: 1e-9,
                    abs_tol: 1e-9,
                },
            )
            .run()?;
    }
    Ok(())
}
