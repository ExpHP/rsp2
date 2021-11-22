use crate::FailResult;
use rsp2_array_types::V3;
use slice_of_array::prelude::*;
use crate::traits::AsPath;

pub fn parse_qpoint(s: &str) -> FailResult<V3> {
    let words: Vec<_> = s.split_ascii_whitespace().collect();

    if words.len() != 3 {
        bail!("Expected 3 whitespace-separated floats or rationals in --qpoint, got {:?}", s);
    }

    words.as_array::<V3<_>>().try_map(|word: &str| {
        if word.contains("/") {
            let mut iter = word.split("/");

            let numer: i32 = iter.next().unwrap().parse()?;
            let denom: i32 = iter.next().unwrap().parse()?;
            if let Some(_) = iter.next() {
                bail!("a rational cannot have multiple '/'!")
            }

            Ok(numer as f64 / denom as f64)
        } else {
            word.parse::<f64>().map_err(|_| {
                // make sure error mentions possibility of using rationals
                format_err!("{:?} is not a valid floating point or rational number", word)
            })
        }
    })
}

pub struct QPointsFile(pub Vec<V3>);

pub fn parse_qpoint_file(s: &str) -> FailResult<Vec<V3>> {
    s.lines()
        .map(|s| {
            let end = s.find("#").unwrap_or(s.len());
            s[..end].trim()
        })
        .filter(|s| !s.is_empty())
        .map(parse_qpoint)
        .collect::<Result<Vec<_>, _>>()
}

impl crate::traits::Load for crate::ui::parse_qpoint::QPointsFile {
    fn load(path: impl AsPath) -> FailResult<Self> {
        let text = path_abs::FileRead::open(path.as_path())?.read_string()?;
        Ok(QPointsFile(parse_qpoint_file(&text)?))
    }
}
