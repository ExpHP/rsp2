extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate serde_json;

extern crate failure;
extern crate itertools;

#[macro_use] extern crate rsp2_assert_close;
extern crate rsp2_integration_test;

use ::rsp2_integration_test::CliTest;

extern crate path_abs;
extern crate rsp2_structure;

use ::path_abs::{PathAbs};

mod shared;
use self::shared::filetypes;

#[test]
#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
fn gnr_test() {
    let abs = |path: &str| PathAbs::new(path).unwrap();
    CliTest::cargo_binary("rsp2")
        .arg("-c").arg(abs("tests/resources/defaults.yaml").as_path())
        .arg("-c").arg(abs("tests/resources/sparse-gnr.yaml").as_path())
        .arg(abs("tests/resources/POSCAR_GNR1.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::Dynmat>(
            "out/gamma-dynmat.json".as_ref(),
            "tests/resources/sparse-gnr-out/gamma-dynmat.json".as_ref(),
            filetypes::DynmatTolerances {
                // The results here seem highly nondeterministic.
                // This seems to be because it at some point needs to relax along eigenvectors,
                // and eigsh uses a randomly initialized starting vector.
                rel_tol: 1e-4,
                abs_tol: 1e-1,
            },
        )
        .run().unwrap();
}
