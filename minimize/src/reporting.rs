use ::std::fmt::Write;
use ::std::fmt;

fn fmt_bins<W: Write>(mut w: W, width: usize, bins: &[u64]) -> Result<(), fmt::Error> {
    let alphabet: Vec<char> = "0123456789abcdefghijklmnopqrstuvwxyz".chars().collect();

    let sparse: Vec<(usize, u64)> = bins.iter().cloned()
        .enumerate().filter(|&(_, x)| x != 0)
        .collect();
    let total: u64 = sparse.iter().map(|&(_,x)| x).sum();

    let mut ceils: Vec<u64> = sparse.iter()
        .map(|&(_, c)| (c * width as u64) as f64 / total as f64) // fill fractions
        .map(|f| f.ceil() as u64)
        .collect();

    let over_amount = ceils.iter().sum::<u64>() - width as u64;
    // FIXME should remove counts according to their actual distribution;
    //  i.e. we should maximize the unit-vector-dot-product of removed counts and over_amount
    let biggest = argmax(ceils.iter()).unwrap();
    ceils[biggest] = ceils[biggest].checked_sub(over_amount).unwrap(); // FIXME

    for (&(i,_), n) in sparse.iter().zip(ceils) {
        let c = alphabet[i % alphabet.len()];
        for _ in 0..n {
            write!(&mut w, "{}", c)?;
        }
    }
    Ok(())
}

fn argmax<T, I>(iter: I) -> Option<usize>
where
    T: Ord, I: IntoIterator<Item=T>,
    T: Clone, // FIXME HACK
{ iter.into_iter().enumerate().max_by_key(|&(_, ref v)| v.clone()).map(|(i,_)| i) }

pub struct Bins<T> {
    check_min: Option<T>,
    check_max: Option<T>,
    divs: Vec<T>,
    bins: Vec<u64>,
}

impl<T:PartialOrd> Bins<T> {
    pub fn new(mut divs: Vec<T>) -> Bins<T>
    {
        assert!(divs.len() >= 2);
        for (a,b) in divs.iter().zip(&divs[1..]) {
            assert!(a < b);
        }
        let check_min = Some(divs.remove(0));
        let check_max = Some(divs.pop().unwrap());
        let bins = vec![0; divs.len() + 1];
        Bins { check_min, check_max, bins, divs }
    }

    pub fn from_iter<I>(divs: Vec<T>, it: I) -> Bins<T>
    where I: IntoIterator<Item=T>,
    {
        let mut bins = Bins::new(divs);
        bins.extend(it);
        bins
    }

    pub fn display(&self) -> Display { Display(&self.bins) }

    pub fn as_counts(&self) -> &[u64] { &self.bins }
}

impl<T> Extend<T> for Bins<T>
where T: PartialOrd
{
    fn extend<I>(&mut self, it: I)
    where I: IntoIterator<Item=T>
    {
        'outer: for x in it {
            if let Some(ref min) = self.check_min { assert!(min <= &x); }
            if let Some(ref max) = self.check_max { assert!(&x <= max); }
            for (i, div) in self.divs.iter().enumerate() {
                if &x < div {
                    self.bins[i] += 1;
                    continue 'outer;
                }
            }
            *self.bins.last_mut().unwrap() += 1;
        }
    }
}

pub struct Display<'a>(&'a [u64]);
impl<'a> fmt::Display for Display<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let width = f.width().unwrap_or(10);
        fmt_bins(f, width, &self.0)
    }
}