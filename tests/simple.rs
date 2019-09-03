#[macro_use]
extern crate rsp2_assert_close;

use rsp2_integration_test::{CliTest, filetypes, resource, cli_test, Result};
use rsp2_structure_io::Poscar;
use path_abs::{FileRead, PathOps};
use std::path::Path;

// A single raman output file is used for all of these tests.

// Used by the test whose output is saved in the resources dir.
const PRECISE_RAMAN_TOL: filetypes::RamanJsonTolerances = filetypes::RamanJsonTolerances {
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
};

// Used by tests that ideally should produce output nearly identical to the PRECISE_RAMAN_TOL
// test, but which may be subject to very minor roundoffs (~1e-11) that may accumulate.
const MODERATE_RAMAN_TOL: filetypes::RamanJsonTolerances = filetypes::RamanJsonTolerances {
    frequency: filetypes::FrequencyTolerances {
        max_acoustic: 0.01,
        rel_tol: 1e-7,
    },
    intensity_nonzero_thresh: 1e-19,
    intensity_nonzero_rel_tol: 1e-6,
};

// Used by tests that differ in ways from the PRECISE_RAMAN_TOL that can lead to
// numerically significant (>1e-7), but not physically different results.
const WEAK_RAMAN_TOL: filetypes::RamanJsonTolerances = filetypes::RamanJsonTolerances {
    frequency: filetypes::FrequencyTolerances {
        max_acoustic: 0.01,
        rel_tol: 1e-4,
    },
    intensity_nonzero_thresh: 1e-12,
    intensity_nonzero_rel_tol: 1e-4,
};

// Uses the rust reimplementations of REBO/KCZ
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test_rust() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("simple-rust.yaml"))
        .arg(resource("simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            resource("simple-out/raman.json").as_ref(),
            PRECISE_RAMAN_TOL,
        )
        .run()
}

#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test_lammps() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("simple-lammps.yaml"))
        .arg(resource("simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            resource("simple-out/raman.json").as_ref(),
            MODERATE_RAMAN_TOL,
        )
        .run()
}

// Tests lattice optimization during relaxation.
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test_optimize() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("simple-relax-optimize.yaml"))
        .arg(resource("simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            resource("simple-out/raman.json").as_ref(),
            WEAK_RAMAN_TOL,
        )
        .check(|dir| Ok({
            let before_cg = read_poscar(dir.join("out/initial.structure/POSCAR"))?;
            assert_close!(
                rel=1e-5,
                before_cg.coords.lattice().norms(),
                [2.33630723, 2.3363072, 13.374096],
            );

            let after_cg = read_poscar(dir.join("out/ev-loop-01.1.structure/POSCAR"))?;
            assert_close!(
                rel=1e-5,
                after_cg.coords.lattice().norms(),
                [2.4592708, 2.4592708, 13.374096],
            );
        }))
        .run()
}

// Tests lattice optimization prior to relaxation.
//
// Uses layers.yaml format so layer sep can also be optimized.
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test_scale_ranges() -> Result<()> {
    let env = cli_test::Environment::init();
    CliTest::cargo_binary(&env, "rsp2")
        .arg("-c").arg(resource("defaults.yaml"))
        .arg("-c").arg(resource("simple-scale-ranges.yaml"))
        .arg(resource("simple.layers.yaml").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            resource("simple-out/raman.json").as_ref(),
            WEAK_RAMAN_TOL,
        )
        .check(|dir| Ok({
            let before_cg = read_poscar(dir.join("out/initial.structure/POSCAR"))?;
            assert_close!(
                rel=1e-5,
                before_cg.coords.lattice().norms(),
                [2.4592708, 2.4592708, 13.374096],
            );

            let after_cg = read_poscar(dir.join("out/ev-loop-01.1.structure/POSCAR"))?;
            assert_eq!(
                before_cg.coords.lattice(),
                after_cg.coords.lattice(),
            );
        }))
        .run()
}

fn read_poscar(path: impl AsRef<Path>) -> Result<Poscar> {
    Ok(Poscar::from_reader(FileRead::open(path)?)?)
}
