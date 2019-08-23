use ::rsp2_integration_test::{CliTest, filetypes, resource, cli_test, Result};
use std::path::Path;

// The workflow to update the test outputs is described in tests/dynmat-at-q.rs.

fn check_001_0_a_local_normals(
    env: &cli_test::Environment,
    expected_outfile: &Path,
) -> Result<()> {
    CliTest::cargo_binary(env, "rsp2-dynmat-at-q")
        .arg("-c").arg(resource("dynmat-at-q/settings.yaml"))
        .arg("-c").arg(resource("dynmat-at-q/kc-local-normals.yaml"))
        .arg(resource("dynmat-at-q/001-0-a.structure"))
        .arg("--qpoint").arg("0 0 0")
        .arg("-o").arg("dynmat.npz")
        .check_file::<filetypes::Dynmat>(
            "dynmat.npz".as_ref(),
            expected_outfile.as_ref(),
            filetypes::DynmatTolerances {
                rel_tol: 1e-5,
                abs_tol: 1e-7,
            },
        )
        .run()
}

#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn local_normals() {
    let env = cli_test::Environment::init();

    check_001_0_a_local_normals(&env, &resource("dynmat-at-q/001-0-a-local-normals.dynmat.npz")).unwrap();
    //check_001_0_a_local_normals(&env, &resource("dynmat-at-q/001-0-a-gamma.dynmat.npz")).unwrap();
}

// makes sure tolerances are not too large
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
#[should_panic(expected = "not nearly equal")]
fn local_normals_specificity() {
    let env = cli_test::Environment::init();

    check_001_0_a_local_normals(&env, &resource("dynmat-at-q/001-0-a-gamma.dynmat.npz")).unwrap();
}
