extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate serde_json;

extern crate path_abs;
extern crate failure;

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
            assert_eq!(actual, expected);
            Ok(())
        })
        .run().unwrap();
}
