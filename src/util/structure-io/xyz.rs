use ::Result;
use ::std::io::prelude::*;

use ::rsp2_structure::{Element, ElementStructure};

/// Writes an XYZ frame to an open file.
///
/// You can freely call this multiple times on the same file
/// to write an animation, since XYZ animations are simply
/// concatenated XYZ files.
pub fn dump<W>(w: W, title: &str, structure: &ElementStructure) -> Result<()>
where W: Write,
{
    _dump(w, title, &structure.to_carts(), structure.metadata())
}

fn _dump<W>(mut w: W, title: &str, carts: &[[f64; 3]], types: &[Element]) -> Result<()>
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
