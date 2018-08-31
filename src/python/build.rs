extern crate walkdir;
use walkdir::WalkDir;
use std::ffi::OsStr;

fn main() {
    // Unfortunately, the `include_dir!` macro seems to work in such a way that
    // rustc does not track the directory for changes, and thus the crate is not
    // rebuilt when it ought to be.
    //
    // This build script serves as a workaround.  When the directory changes,
    // rust is told to rerun this build script, which then forces it to rebuild
    // the crate.
    for entry in WalkDir::new("rsp2").into_iter().filter_map(|e| e.ok()) {
        // ...it is also outrageously memory hungry.  We'll want to make sure
        // nothing unexpected is in the directory!
        if entry.file_type().is_file() {
            let ext = entry.path().extension();
            if !(ext == Some(OsStr::new("py")) || ext == Some(OsStr::new("pyc"))) {
                panic!("{}: Unexpected file in python directory!", entry.path().display())
            }
        }
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
}
