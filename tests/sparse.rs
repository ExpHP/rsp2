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
extern crate rsp2_python;
extern crate rsp2_fs_util;

use ::path_abs::{PathAbs};

mod shared;
use self::shared::filetypes;

#[ignore] // This test is expensive; use `cargo test -- --ignored` to run it!
#[test]
fn simple_test() {
    let abs = |path: &str| PathAbs::new(path).unwrap();
    CliTest::cargo_binary("rsp2")
        .arg("-c").arg(abs("tests/resources/defaults.yaml").as_path())
        .arg("-c").arg(abs("tests/resources/sparse.yaml").as_path())
        .arg(abs("tests/resources/simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::Dynmat>(
            "out/gamma-dynmat.npz".as_ref(),
            "tests/resources/sparse-out/gamma-dynmat.npz".as_ref(),
            filetypes::DynmatTolerances {
                rel_tol: 1e-9,
                abs_tol: 1e-9,
            },
        )
        .run().unwrap();
}
