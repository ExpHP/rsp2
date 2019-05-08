use ::rsp2_integration_test::{CliTest, filetypes, resource, cli_test};

#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test() {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("simple.yaml"))
        .arg(resource("simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            resource("simple-out/raman.json").as_ref(),
            filetypes::RamanJsonTolerances {
                frequency: filetypes::FrequencyTolerances {
                    max_acoustic: 0.01,
                    rel_tol: 1e-9,
                },
                // the zero thresh can be pressed down surprisingly far;
                // right now I'm not seeing failures until it is 1e-24.
                //
                // This value is probably pushing my luck, but I just want to satisfy
                // my curiosity and see how many software stack updates it takes before
                // this fails.
                intensity_nonzero_thresh: 1e-19,
                intensity_nonzero_rel_tol: 1e-9,
            },
        )
        .run().unwrap();
}

// Uses the rust reimplementations of REBO/KCZ
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test_rust() {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("simple-rust.yaml"))
        .arg(resource("simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            resource("simple-out/raman.json").as_ref(),
            filetypes::RamanJsonTolerances {
                frequency: filetypes::FrequencyTolerances {
                    max_acoustic: 0.01,
                    rel_tol: 1e-7,
                },
                intensity_nonzero_thresh: 1e-19,
                intensity_nonzero_rel_tol: 1e-6,
            },
        )
        .run().unwrap();
}

// Tests optimization during relaxation.
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test_optimize() {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("simple-relax-optimize.yaml"))
        .arg(resource("simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            resource("simple-out/raman.json").as_ref(),
            filetypes::RamanJsonTolerances {
                frequency: filetypes::FrequencyTolerances {
                    max_acoustic: 0.01,
                    rel_tol: 1e-4,
                },
                intensity_nonzero_thresh: 1e-19,
                intensity_nonzero_rel_tol: 1e-4,
            },
        )
        .run().unwrap();
}
