
use ::Result;

use ::std::io::prelude::*;
use ::itertools::Itertools;

use ::rsp2_structure::{Element, ElementStructure};
use ::rsp2_structure::{Lattice, CoordsKind};
use ::rsp2_array_types::{Envee, Unvee};

use ::vasp_poscar::{Poscar, RawPoscar, ScaleLine};

/// Writes a POSCAR to an open file.
pub fn dump<W>(
    mut w: W,
    title: &str,
    structure: &ElementStructure,
) -> Result<()>
where W: Write
{
    // FIXME replace with e.g. Poscar::set_site_symbols when available
    let mut group_counts = vec![];
    let mut group_symbols = vec![];
    for (key, group) in &structure.metadata().iter().group_by(|typ| *typ) {
        group_counts.push(group.count());
        group_symbols.push(key.symbol().into());
    }

    // FIXME use Poscar builder when available
    let poscar = RawPoscar {
        comment: title.into(),
        scale: ScaleLine::Factor(1.0),
        lattice_vectors: structure.lattice().matrix().into_array(),
        positions: ::vasp_poscar::Coords::Cart(structure.to_carts().unvee()),
        group_symbols: Some(group_symbols),
        group_counts,
        velocities: None,
        dynamics: None,
    }.validate().map_err(|e| {
        let e: ::vasp_poscar::failure::Error = e.into(); e.compat()
    })?;

    write!(w, "{}", poscar)?;

    Ok(())
}

// FIXME This probably shouldn't exist.
/// Reads a POSCAR from an open file.
///
/// This forcibly reads to EOF because it must construct a BufReader.
pub fn load<R>(mut f: R) -> Result<ElementStructure>
where R: Read,
{
    let out = load_txt(::std::io::BufReader::new(&mut f))?;
    f.read_to_end(&mut vec![])?;
    Ok(out)
}

/// Reads a POSCAR from an open file.
pub fn load_txt<R>(f: R) -> Result<ElementStructure>
where R: BufRead,
{
    use vasp_poscar::failure::ResultExt;
    let poscar = Poscar::from_reader(f).compat()?;
    let RawPoscar {
        scale, lattice_vectors, positions,
        group_symbols, group_counts,
        ..
    } = poscar.raw();

    // FIXME use Poscar::scaled_lattice_vectors() once it is available
    // FIXME use Poscar::scaled_cart_positions() once it is available
    assert_eq!(scale, ScaleLine::Factor(1.0));
    let lattice = Lattice::from(&lattice_vectors);
    let coords = match positions {
        ::vasp_poscar::Coords::Cart(p) => CoordsKind::Carts(p.envee()),
        ::vasp_poscar::Coords::Frac(p) => CoordsKind::Fracs(p.envee()),
    };

    let group_symbols = group_symbols.expect("symbols are required").into_iter()
        .map(|sym| match Element::from_symbol(&sym) {
            None => bail!("Unknown element: '{}'", sym),
            Some(e) => Ok(e),
        })
        .collect::<Result<Vec<Element>>>()?;

    // FIXME use Poscar method once available
    let elements = izip!(group_counts, group_symbols)
        .flat_map(|(c, sym)| ::std::iter::repeat(sym).take(c))
        .collect::<Vec<_>>();

    assert_eq!(elements.len(), coords.len());
    Ok(ElementStructure::new(lattice, coords, elements))
}
