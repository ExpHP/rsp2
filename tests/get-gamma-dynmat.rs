use ::rsp2_integration_test::{CliTest, filetypes, resource, cli_test, Result};

#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn dynmat_lammps() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("gamma-dynmat-lammps.yaml"))
        .arg(resource("001-a-relaxed-kcz.vasp"))
        .arg("-o").arg("out")
        .check_file::<filetypes::Dynmat>(
            "out/gamma-dynmat.npz".as_ref(),
            resource("gamma-dynmat-out/gamma-dynmat-lammps.npz").as_ref(),
            filetypes::DynmatTolerances {
                rel_tol: 1e-9,
                abs_tol: 1e-9,
            },
        )
        .run()
}

// Test custom DispFns on the potentials implemented in Rust
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn dynmat_rust() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("gamma-dynmat-rust.yaml"))
        .arg(resource("001-a-relaxed-kcz.vasp"))
        .arg("-o").arg("out")
        .check_file::<filetypes::Dynmat>(
            "out/gamma-dynmat.npz".as_ref(),
            resource("gamma-dynmat-out/gamma-dynmat-rust.npz").as_ref(),
            filetypes::DynmatTolerances {
                rel_tol: 1e-9,
                abs_tol: 1e-9,
            },
        )
        .run()
}

#[test]
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
fn gnr_test() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("gnr.yaml"))
        .arg(resource("POSCAR_GNR1.vasp"))
        .arg("-o").arg("out")
        .check_file::<filetypes::Dynmat>(
            "out/gamma-dynmat.npz".as_ref(),
            resource("gamma-dynmat-out/gamma-dynmat-gnr.npz").as_ref(),
            filetypes::DynmatTolerances {
                rel_tol: 1e-9,
                abs_tol: 1e-9,
            },
        )
        .run()
}
