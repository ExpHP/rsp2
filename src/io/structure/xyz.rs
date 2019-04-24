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
use std::io::{Lines};

use rsp2_structure::{Element};

use rsp2_array_types::V3;

//--------------------------------------------------------------------------------------
// public API

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz<
    Title = String,
    Carts = Vec<V3>,
    Elements = Vec<Element>,
> {
    pub title: Title,
    pub carts: Carts,
    pub elements: Elements,
}

impl<Title, Carts, Elements> Xyz<Title, Carts, Elements>
where
    Title: AsRef<str>,
    Carts: AsRef<[V3]>,
    Elements: AsRef<[Element]>,
{
    /// Writes an XYZ frame to an open file.
    ///
    /// You can freely call this multiple times on the same file
    /// to write an animation, since XYZ animations are simply
    /// concatenated XYZ files.
    pub fn to_writer(&self, mut w: impl Write) -> FailResult<()> {
        dump(&mut w, self.title.as_ref(), self.carts.as_ref(), self.elements.as_ref())
    }
}

impl Xyz {
    /// Read a single-frame XYZ file.
    pub fn from_buf_reader(r: impl BufRead) -> FailResult<Self> {
        let mut frames = Self::anim_from_buf_reader(r)?;
        match frames.len() {
            0 => bail!("XYZ file is empty!"),
            1 => Ok(frames.remove(0)),
            _ => {
                warn!("\
                    An animation was supplied where a single-frame XYZ file was expected! \
                    Only the first frame will be used.\
                ");
                Ok(frames.remove(0))
            },
        }
    }

    /// Read a multiple-frame XYZ file.
    pub fn anim_from_buf_reader(mut r: impl BufRead) -> FailResult<Vec<Self>> {
        let r: &mut dyn BufRead = &mut r;
        load(r.lines())
    }
}

//--------------------------------------------------------------------------------------
// implementation

fn dump(w: &mut dyn Write, title: &str, carts: &[V3], types: &[Element]) -> FailResult<()>
{
    assert!(!title.contains("\n"));
    assert!(!title.contains("\r"));
    assert_eq!(carts.len(), types.len());

    writeln!(w, "{}", carts.len())?;
    writeln!(w, "{}", title)?;
    for (V3([x, y, z]), typ) in carts.iter().zip(types) {
        writeln!(w, " {:>2} {} {} {}", typ.symbol(), x, y, z)?;
    }

    Ok(())
}

fn load(mut r: Lines<&mut dyn BufRead>) -> FailResult<Vec<Xyz>> {
    let mut out = vec![];
    while let Some(frame) = load_frame_or_eof(&mut r)? {
        out.push(frame);
    }
    Ok(out)
}

fn load_frame_or_eof(r: &mut Lines<&mut dyn BufRead>) -> FailResult<Option<Xyz>>
{
    let count = match r.next() {
        None => return Ok(None), // eof
        Some(line) => line?.trim().parse::<usize>()?,
    };
    let title = r.next().ok_or_else(|| format_err!("unexpected EOF!"))??;

    let mut elements = Vec::with_capacity(count);
    let mut carts = Vec::with_capacity(count);
    for read_so_far in 0..count {
        let line = r.next().ok_or_else(|| {
            format_err!("unexpected EOF! (expected {} atoms, found {})", count, read_so_far)
        })??;

        let mut words = line.split_whitespace();
        elements.push(Element::from_symbol(words.next().ok_or_else(|| {
            format_err!("unexpected empty line when reading atom from XYZ file")
        })?)?);

        carts.push({
            V3::try_from_fn(|_| Ok::<_, failure::Error>({
                words.next()
                    .ok_or_else(|| format_err!("expected 3 coordinates after species name"))?
                    .parse()?
            }))?
        });

        if let Some(_) = words.next() {
            warn_once!{"\
                Extra junk (possibly connectivity info?) found after the coordinates \
                in an XYZ file.  These will be ignored by RSP2.\
            "}
        }
    }

    Ok(Some(Xyz { title, carts, elements }))
}
