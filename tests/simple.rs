extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate serde_json;

extern crate failure;

#[macro_use] extern crate rsp2_assert_close;
extern crate rsp2_integration_test;

use ::rsp2_integration_test::CliTest;

extern crate path_abs;
use ::path_abs::{PathAbs};

mod shared;
use self::shared::filetypes;

#[test]
fn simple_test() {
    let abs = |path: &str| PathAbs::new(path).unwrap();
    CliTest::cargo_binary("rsp2")
        .arg("-c").arg(abs("tests/resources/simple.yaml").as_path())
        .arg(abs("tests/resources/simple.vasp").as_path())
        .arg("-o").arg("out")
        .check_file::<filetypes::RamanJson>(
            "out/raman.json".as_ref(),
            "tests/resources/simple-out/raman.json".as_ref(),
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
