
use ::FailResult;

use ::std::io::prelude::*;
use ::itertools::Itertools;

use ::rsp2_structure::{Element, ElementStructure};
use ::rsp2_structure::{Lattice, CoordsKind};
use ::rsp2_array_types::{Envee, Unvee};

use ::vasp_poscar::{Poscar, RawPoscar, ScaleLine};

/// Writes a POSCAR to an open file.
pub fn dump(
    mut w: impl Write,
    title: &str,
    structure: &ElementStructure,
) -> FailResult<()>
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
pub fn load(mut f: impl Read) -> FailResult<ElementStructure>
{
    let out = load_txt(::std::io::BufReader::new(&mut f))?;
    f.read_to_end(&mut vec![])?;
    Ok(out)
}

/// Reads a POSCAR from an open file.
pub fn load_txt(f: impl BufRead) -> FailResult<ElementStructure>
{
    use vasp_poscar::failure::ResultExt;
    let poscar = Poscar::from_reader(f).compat()?;
    let RawPoscar {
        scale, lattice_vectors, positions,
        group_symbols, group_counts, comment,
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

    let group_elems = {
        // we need symbols, but prior to VASP 5 they were not even part of
        // the format, so some programs don't write them where they belong.
        // Sometimes they are used as the title comment (by phonopy, ASE...).
        let group_symbols = group_symbols.unwrap_or_else(|| {
            let symbols = comment.split_whitespace().map(|s| s.to_string()).collect::<Vec<_>>();
            assert_eq!(
                symbols.len(), group_counts.len(),
                "Symbols must be given either in the standard location or the POSCAR comment.",
            );
            // pray for the best.  If they're not symbols, it is at least unlikely
            // that the next step will erroneously "succeed"
            symbols
        });
        group_symbols.into_iter()
            .map(|sym| match Element::from_symbol(&sym) {
                None => bail!("Unknown element: '{}'", sym),
                Some(e) => Ok(e),
            })
            .collect::<FailResult<Vec<Element>>>()?
    };

    // FIXME use Poscar method once available
    let elements = izip!(group_counts, group_elems)
        .flat_map(|(c, elem)| ::std::iter::repeat(elem).take(c))
        .collect::<Vec<_>>();

    assert_eq!(elements.len(), coords.len());
    Ok(ElementStructure::new(lattice, coords, elements))
}
