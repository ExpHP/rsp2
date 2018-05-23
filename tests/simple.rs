extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate serde_json;

extern crate path_abs;
extern crate failure;

#[macro_use] extern crate rsp2_assert_close;
extern crate rsp2_integration_test;

use ::rsp2_integration_test::CliTest;

use ::path_abs::{FileRead, PathAbs};
use ::std::path::Path;
use ::failure::Error;

mod shared;
use self::shared::filetypes;

#[test]
fn simple_test() {
    let abs = |path: &str| PathAbs::new(path).unwrap();
    CliTest::cargo_binary("rsp2")
        .arg("-c").arg(abs("tests/resources/simple.yaml").as_path())
        .arg(abs("tests/resources/simple.vasp").as_path())
        .arg("-o").arg("out")
        .check(|dir| {
            let read = |path: &Path| Ok::<_, Error>(::serde_json::from_reader(FileRead::read(path)?)?);
            let actual: filetypes::RamanJson = read(&dir.join("out/raman.json"))?;
            let expected: filetypes::RamanJson = read("tests/resources/simple-out/raman.json".as_ref())?;

            // Comparing this data in a meaningful manner is not easy; there is no
            // one-size-fits-all tolerance.

            // Acoustic mode frequencies; magnitude is irrelevant as long as it is not large.
            actual.check_against(&expected, filetypes::RamanJsonTolerances {
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
            });
            Ok(())
        })
        .run().unwrap();
}
