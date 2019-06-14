// void output_xml(string filename, const vector<double> &gradient)
use crate::traits::{AsPath, Save};
use crate::FailResult;
use rsp2_array_types::V3;
use std::io::prelude::*;
use path_abs::FileWrite;

pub struct FakeVasprun<V = Vec<V3>> {
    pub force: V,
}

impl<V: AsRef<[V3]>> Save for FakeVasprun<V> {
    fn save(&self, path: impl AsPath) -> FailResult<()> {
        dump(FileWrite::create(path.as_path())?, self.force.as_ref())
    }
}

fn dump<W: Write>(mut w: W, force: &[V3]) -> FailResult<()> {
    write!(w, "{}", r#"
<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <generator>
  <i name="program" type="string">vasp</i>
  <i name="version" type="string">5.4.1</i>
 </generator>
 <calculation>
  <varray name="forces">
"#.trim_start())?;

    for V3([x, y, z]) in force {
        writeln!(w, "   <v> {} {} {}</v>", x, y, z)?;
    }

    write!(w, "{}", r#"
  </varray>
 </calculation>
</modeling>
"#.trim_start())?;

    Ok(())
}
