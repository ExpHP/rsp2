/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

use crate::FailResult;

use std::io::prelude::*;
use std::borrow::Borrow;

use rsp2_structure::{Element, Coords as Coords, Lattice, CoordsKind};
use rsp2_array_types::{Envee, Unvee};

use vasp_poscar as imp;

//--------------------------------------------------------------------------------------
// public API

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poscar<
    Comment = String,
    Coord = Coords,
    Elements = Vec<Element>,
> {
    pub comment: Comment,
    pub coords: Coord,
    pub elements: Elements,
}

impl<Comment, Coord, Elements> Poscar<Comment, Coord, Elements>
where
    Comment: AsRef<str>,
    Coord: Borrow<Coords>,
    Elements: AsRef<[Element]>,
{
    /// Writes a POSCAR to an open file.
    pub fn to_writer(&self, mut w: impl Write) -> FailResult<()> {
        dump(&mut w, self.comment.as_ref(), self.coords.borrow(), self.elements.as_ref())
    }
}

impl Poscar {
    // FIXME This probably shouldn't exist.
    /// Reads a POSCAR from an open file.
    ///
    /// This forcibly reads to EOF because it must construct a BufReader.
    pub fn from_reader(mut f: impl Read) -> FailResult<Self> {
        let out = load_txt(&mut std::io::BufReader::new(&mut f))?;
        f.read_to_end(&mut vec![])?;
        Ok(out)
    }

    /// Reads a POSCAR from an open file.
    pub fn from_buf_reader(f: impl BufRead) -> FailResult<Self> {
        load_txt(&mut std::io::BufReader::new(f))
    }
}

//--------------------------------------------------------------------------------------
// implementation

// monomorphic
fn dump(
    w: &mut dyn Write,
    title: &str,
    coords: &Coords,
    elements: &[Element],
) -> FailResult<()>
{
    write!(w, "{}",
        vasp_poscar::Builder::new()
            .comment(title)
            .lattice_vectors(coords.lattice().matrix().as_array())
            .positions(vasp_poscar::Coords::Cart(coords.to_carts().unvee()))
            .site_symbols(elements.iter().map(|&elem| elem.symbol()))
            .build()?,
    )?;
    Ok(())
}

/// Reads a POSCAR from an open file.
fn load_txt(f: &mut dyn BufRead) -> FailResult<Poscar>
{
    use vasp_poscar::failure::ResultExt;
    let poscar = imp::Poscar::from_reader(f).compat()?;

    let comment = poscar.comment().to_string();
    let lattice = Lattice::from(&poscar.scaled_lattice_vectors());
    let coords = match poscar.scaled_positions() {
        imp::Coords::Cart(p) => CoordsKind::Carts(p.to_vec().envee()),
        imp::Coords::Frac(p) => CoordsKind::Fracs(p.to_vec().envee()),
    };

    let group_elems = {
        // we need symbols, but prior to VASP 5 they were not even part of
        // the format, so some programs don't write them where they belong.
        // Sometimes they are used as the title comment (by phonopy, ASE...).
        let group_symbols = match poscar.group_symbols() {
            Some(iter) => iter.map(|s| s.to_string()).collect(),
            None => {
                let words = poscar.comment().split_whitespace();
                let symbols = words.map(|s| s.to_string()).collect::<Vec<_>>();
                ensure!(
                    symbols.len() == poscar.group_counts().len(),
                    "Symbols must be given either in the standard location or the POSCAR comment."
                );
                // pray for the best.  If they're not the group symbols, it is at least
                // unlikely that the next step will erroneously "succeed"
                symbols
            },
        };
        group_symbols.into_iter()
            .map(|sym| Element::from_symbol(&sym))
            .collect::<Result<Vec<Element>, _>>()?
    };

    let elements = zip_eq!(poscar.group_counts(), group_elems)
        .flat_map(|(c, elem)| std::iter::repeat(elem).take(c))
        .collect::<Vec<_>>();

    assert_eq!(elements.len(), coords.len());
    let coords = Coords::new(lattice, coords);
    Ok(Poscar { comment, coords, elements })
}
