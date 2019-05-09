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

use rsp2_array_types::{V3};
use rsp2_structure::{Coords, Element};

use num_complex::Complex64;
use std::borrow::Borrow;
use std::io::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VSimAscii<
    Comment = String,
    Coord = Coords,
    Elements = Vec<Element>,
    Metadata = AsciiMetadata,
> {
    /// The first line of the file.
    pub comment: Comment,
    pub coords: Coord,
    pub elements: Elements,
    pub metadata: Metadata,
}

impl<Comment, Coord, Elements, Metadata> VSimAscii<Comment, Coord, Elements, Metadata>
where
    Comment: AsRef<str>,
    Coord: Borrow<Coords>,
    Elements: AsRef<[Element]>,
    Metadata: Borrow<AsciiMetadata>,
{
    /// Writes a .ascii to an open file.
    pub fn to_writer(&self, mut w: impl Write) -> FailResult<()> {
        dump(
            &mut w,
            self.comment.as_ref(),
            self.coords.borrow(),
            self.elements.as_ref(),
            self.metadata.borrow(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AsciiMetadata {
    phonons: Vec<Phonon>,
    labels: Option<Vec<String>>,
}

impl AsciiMetadata {
    pub fn new() -> Self { Default::default() }

    pub fn clear_labels(&mut self) -> &mut Self
    { self.labels = None; self }

    pub fn set_labels<Ss, S>(&mut self, labels: Ss) -> &mut Self
    where
        Ss: IntoIterator<Item=S>,
        S: Into<String>,
    {
        self.labels = Some(labels.into_iter().map(|s| s.into()).collect());
        self
    }

    pub fn add_phonon(&mut self, phonon: Phonon) -> &mut Self
    { self.phonons.push(phonon); self }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Phonon {
    pub qpoint_frac: V3,

    /// Wavenumber (cm^-1).
    ///
    /// Only used for labelling purposes in `V_Sim`.
    pub energy: f64,

    /// Displacement directions in Cartesian.
    ///
    /// Scale doesn't matter; `V_Sim` will rescale it.
    pub displacements: Vec<V3<Complex64>>,
}

fn dump(
    w: &mut dyn Write,
    title: &str,
    coords: &Coords,
    elements: &[Element],
    metadata: &AsciiMetadata,
) -> FailResult<()> {
    if title.contains('\r') || title.contains('\n') {
        bail!("V_Sim ascii header comment cannot contain newline.")
    }

    assert_eq!(coords.len(), elements.len());
    if let Some(labels) = metadata.labels.as_ref() {
        assert_eq!(coords.len(), labels.len());
    }

    // Rotate to lower triangular
    let &[
        [dxx,  _0,  _1],
        [dyx, dyy,  _2],
        [dzx, dzy, dzz],
    ] = coords.lattice().rotate_to_lower_triangular().matrix().as_array();
    let fracs = coords.to_fracs();

    // First three lines have a fairly rigid format
    writeln!(w, "{}", title)?;
    writeln!(w, " {:e} {:e} {:e}", dxx, dyx, dyy)?;
    writeln!(w, " {:e} {:e} {:e}", dzx, dzy, dzz)?;

    // Rest can be reordered however.
    writeln!(w, "#keyword: reduced")?; // coords are fractional

    let AsciiMetadata { labels, phonons } = metadata;
    for (i, (V3([fa, fb, fc]), &elem)) in zip_eq!(fracs, elements).enumerate() {
        if let Some(labels) = labels {
            writeln!(w, " {:e} {:e} {:e} {} {}", fa, fb, fc, elem.symbol(), &labels[i])?;
        } else {
            writeln!(w, " {:e} {:e} {:e} {}", fa, fb, fc, elem.symbol())?;
        }
    }

    for &Phonon { qpoint_frac, energy, ref displacements } in phonons {
        let V3([kx, ky, kz]) = qpoint_frac;

        writeln!(w, "#metaData: qpt=[{:e};{:e};{:e}; {:e} \\", kx, ky, kz, energy)?;
        assert_eq!(displacements.len(), coords.len());

        for V3([x, y, z]) in displacements {
            write!(w, "#")?;
            write!(w, "; {:e}; {:e}; {:e}", x.re, y.re, z.re)?;
            write!(w, "; {:e}; {:e}; {:e}", x.im, y.im, z.im)?;
            writeln!(w, " \\")?;
        }
        writeln!(w, "#]")?;
    }

    Ok(())
}
