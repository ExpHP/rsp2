use ::rsp2_integration_test::{CliTest, filetypes, resource, cli_test, Result};

#[test]
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
fn gnr_test() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("sparse-gnr.yaml"))
        .arg(resource("POSCAR_GNR1.vasp"))
        .arg("-o").arg("out")
        .check_file::<filetypes::Dynmat>(
            "out/gamma-dynmat.npz".as_ref(),
            resource("sparse-gnr-out/gamma-dynmat.npz").as_ref(),
            filetypes::DynmatTolerances {
                // The results here seem highly nondeterministic.
                // This seems to be because it at some point needs to relax along eigenvectors,
                // and eigsh uses a randomly initialized starting vector.
                rel_tol: 1e-4,
                abs_tol: 1e-1,
            },
        )
        .run()
}
