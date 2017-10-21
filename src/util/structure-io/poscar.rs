
use ::Result;

use ::std::io::prelude::*;
use ::itertools::Itertools;

use ::rsp2_structure::{Element, ElementStructure};
use ::rsp2_structure::{CoordStructure, Lattice, Coords};

// HACK this is closer to how the function was originally written,
//      with support for atom types, but I don't want to have to worry
//      about atom types right now.
/// Writes a POSCAR to an open file.
pub fn dump<W>(
    w: W,
    title: &str,
    structure: &ElementStructure,
) -> Result<()>
where W: Write,
{
    _dump(
        w,
        title,
        &structure.clone().map_metadata(|_| ()),
        structure.metadata(),
    )
}

/// Writes a POSCAR to an open file.
fn _dump<W>(
    mut w: W,
    title: &str,
    structure: &CoordStructure,
    types: &[Element],
) -> Result<()>
where W: Write
{
    assert!(!title.contains("\n"));
    assert!(!title.contains("\r"));
    assert_eq!(structure.num_atoms(), types.len());

    writeln!(&mut w, "{}", title)?;
    writeln!(&mut w, " 1.0")?;
    for row in structure.lattice().matrix().iter() {
        writeln!(&mut w, "  {} {} {}", row[0], row[1], row[2])?;
    }

    {
        let mut pairs = Vec::with_capacity(structure.num_atoms());
        for (key, group) in &types.iter().group_by(|typ| *typ) {
            pairs.push((key, group.count()));
        }
        for &(typ, _) in &pairs { write!(&mut w, " {}", typ.symbol())?; }
        writeln!(&mut w)?;
        for &(_, count) in &pairs { write!(&mut w, " {}", count)?; }
        writeln!(&mut w)?;
    }

    // writeln!(&mut w, "Selective Dynamics")?;
    writeln!(&mut w, "Cartesian")?;

    for (c, typ) in structure.to_carts().iter().zip(types) {
        writeln!(&mut w, " {} {} {} {}", c[0], c[1], c[2], typ.symbol())?;
    }

    Ok(())
}

/// Reads a POSCAR from an open file.
pub fn load<R>(f: R) -> Result<ElementStructure>
where R: Read,
{
    // this is to get us up and running and nothing else
    let f = ::std::io::BufReader::new(f);
    let mut lines = f.lines();

    // title
    {
        let _ = lines.next().unwrap()?;
    }

    // scale
    {
        let s = lines.next().unwrap()?;
        assert_eq!(s.trim().parse::<f64>().unwrap(), 1f64);
    }

    // lattice
    let lattice = &mut [[0f64; 3]; 3];
    {
        let lattice_lines = lines.by_ref().take(3).collect::<Vec<_>>();
        for (i, line) in lattice_lines.into_iter().enumerate() {
            let words = line?.trim().split_whitespace().map(String::from).collect::<Vec<_>>();
            assert_eq!(words.len(), 3);
            for (k, word) in words.into_iter().enumerate() {
                lattice[i][k] = word.parse().unwrap();
            }
        }
    }
    let lattice = Lattice::new(lattice);

    let (elements, n);
    {
        // atom types
        let kinds = {
            lines.next().unwrap()?
                .trim().split_whitespace()
                .map(|sym| match Element::from_symbol(sym) {
                    None => bail!("Unknown element: '{}'", sym),
                    Some(e) => Ok(e),
                })
                .collect::<Result<Vec<Element>>>()?
        };

        // atom counts
        let kounts = {
            lines.next().unwrap()?
                .trim().split_whitespace()
                .map(|s| Ok(s.parse()?))
                .collect::<Result<Vec<usize>>>()?
        };

        n = kounts.iter().sum();
        elements = izip!(kounts, kinds)
            .flat_map(|(c, sym)| ::std::iter::repeat(sym).take(c))
            .collect();
    };


    // selective dynamics and/or cartesian
    let direct = {
        let s = lines.next().unwrap()?;
        match s.chars().next().unwrap() {
            'c' | 'C' | 'k' | 'K' => false,
            'd' | 'D' => true,
            _ => panic!(),
        }
    };

    // coords
    let coords = {
        let coord_lines = lines.by_ref().take(n).collect::<Vec<_>>();
        assert_eq!(coord_lines.len(), n);

        let mut coords: Vec<[f64; 3]> = Vec::with_capacity(n);
        for line in coord_lines {
            let words = line?.trim().split_whitespace().map(String::from).collect::<Vec<_>>();
            // maybe atomic symbol
            assert!([3,4].contains(&words.len()));

            let mut row = [0f64; 3];
            for k in 0..3 {
                row[k] = words[k].parse().unwrap();
            }
            coords.push(row);
        }
        coords
    };

    let coords = match direct {
        true  => Coords::Fracs(coords),
        false => Coords::Carts(coords),
    };

    // we don't support any other junk
    while let Some(line) = lines.next() {
        assert_eq!(line?.trim(), "");
    }

    Ok(ElementStructure::new(lattice, coords, elements))
}
