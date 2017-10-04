use ::std::io;
use ::std::io::prelude::*;
use ::itertools::Itertools;
use ::std::ascii::AsciiExt;

use ::sp2_structure::{CoordStructure, Lattice, Coords};

// HACK this is closer to how the function was originally written,
//      with support for atom types, but I don't want to have to worry
//      about atom types right now.
/// Writes a POSCAR to an open file.
pub fn dump_carbon<W>(
    w: W,
    title: &str,
    structure: &CoordStructure,
) -> io::Result<()>
where
    W: Write,
{
    _dump_carbon(
        w,
        title,
        structure,
        &vec!["S"; structure.num_atoms()],
    )
}

// HACK this is closer to how the function was originally written,
//      with support for atom types, but I don't want to have to worry
//      about atom types right now.
/// Writes a POSCAR to an open file.
fn _dump_carbon<W, S>(
    mut w: W,
    title: &str,
    structure: &CoordStructure,
    types: &[S],
) -> io::Result<()>
where
    W: Write,
    S: ::std::borrow::Borrow<str>,
{
    assert!(!title.contains("\n"));
    assert!(!title.contains("\r"));
    assert!(types.iter().all(|s| s.borrow().chars().all(|c| c.is_ascii() && c.is_alphabetic())));
    assert_eq!(structure.num_atoms(), types.len());

    writeln!(&mut w, "{}", title)?;
    writeln!(&mut w, " 1.0")?;
    for row in structure.lattice().matrix().iter() {
        writeln!(&mut w, "  {} {} {}", row[0], row[1], row[2])?;
    }

    {
        let mut pairs = Vec::with_capacity(structure.num_atoms());
        for (key, group) in &types.iter().group_by(|s| s.borrow()) {
            pairs.push((key, group.count()));
        }
        for &(symbol, _) in &pairs { write!(&mut w, " {}", symbol)?; }
        writeln!(&mut w)?;
        for &(_, count) in &pairs { write!(&mut w, " {}", count)?; }
        writeln!(&mut w)?;
    }

    // writeln!(&mut w, "Selective Dynamics")?;
    writeln!(&mut w, "Cartesian")?;

    for (c, typ) in structure.to_carts().iter().zip(types) {
        writeln!(&mut w, " {} {} {} {}", c[0], c[1], c[2], typ.borrow())?;
    }

    Ok(())
}

/// Reads a POSCAR from an open file.
pub fn load_carbon<R>(f: R) -> io::Result<CoordStructure>
  where
    R: Read,
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
    let mut lattice = [[0f64; 3]; 3];
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

    // atom types
    {
        let s = lines.next().unwrap()?;
        let words = s.trim().split_whitespace().map(String::from).collect::<Vec<_>>();
        assert_eq!(words, vec!["C"]);
    }
    // atom counts
    let n: usize = {
        let s = lines.next().unwrap()?;
        let words = s.trim().split_whitespace().collect::<Vec<_>>();
        assert_eq!(words.len(), 1);
        words[0].parse().unwrap()
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

    Ok(CoordStructure::new_coords(lattice, coords))
}