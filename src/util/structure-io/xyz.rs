use ::FailResult;
use ::std::io::prelude::*;

use ::rsp2_structure::{Element, ElementStructure};

use ::rsp2_array_types::V3;

/// Writes an XYZ frame to an open file.
///
/// You can freely call this multiple times on the same file
/// to write an animation, since XYZ animations are simply
/// concatenated XYZ files.
pub fn dump<W>(w: W, title: &str, structure: &ElementStructure) -> FailResult<()>
where W: Write,
{
    _dump(w, title, &structure.to_carts(), structure.metadata())
}

fn _dump<W>(mut w: W, title: &str, carts: &[V3], types: &[Element]) -> FailResult<()>
where W: Write,
{
    assert!(!title.contains("\n"));
    assert!(!title.contains("\r"));
    assert_eq!(carts.len(), types.len());

    writeln!(&mut w, "{}", carts.len())?;
    writeln!(&mut w, "{}", title)?;
    for (c, typ) in carts.iter().zip(types) {
        writeln!(&mut w, " {:>2} {} {} {}", typ.symbol(), c[0], c[1], c[2])?;
    }

    Ok(())
}
