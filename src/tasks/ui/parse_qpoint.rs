use crate::FailResult;
use rsp2_array_types::V3;
use slice_of_array::prelude::*;

pub fn parse_qpoint(s: &str) -> FailResult<V3> {
    let words: Vec<_> = s.split_ascii_whitespace().collect();

    if words.len() != 3 {
        bail!("Expected 3 whitespace-separated floats or rationals in --qpoint");
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
