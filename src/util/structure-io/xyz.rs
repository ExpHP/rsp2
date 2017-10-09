use ::std::io;
use ::std::io::prelude::*;
use ::std::ascii::AsciiExt;

use ::sp2_structure::CoordStructure;

/// Writes an XYZ frame to an open file.
///
/// You can freely call this multiple times on the same file
/// to write an animation, since XYZ animations are simply
/// concatenated XYZ files.
pub fn dump_carbon<W, S>(w: W, title: &str, structure: &CoordStructure) -> io::Result<()>
  where
    W: Write,
{
    dump_raw(w, title, &structure.to_carts(), &vec!["C"; structure.num_atoms()])
}

fn dump_raw<W, S>(mut w: W, title: &str, carts: &[[f64; 3]], types: &[S]) -> io::Result<()>
  where
    W: Write,
    S: ::std::borrow::Borrow<str>,
{
    assert!(!title.contains("\n"));
    assert!(!title.contains("\r"));
    assert!(types.iter().all(|s| s.borrow().chars().all(|c| c.is_ascii() && c.is_alphabetic())));
    assert_eq!(carts.len(), types.len());

    writeln!(&mut w, "{}", carts.len())?;
    writeln!(&mut w, "{}", title)?;
    for (c, typ) in carts.iter().zip(types) {
        writeln!(&mut w, " {:>2} {} {} {}", typ.borrow(), c[0], c[1], c[2])?;
    }

    Ok(())
}
