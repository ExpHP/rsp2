/* ********************************************************************** **
**  This file is part of rsp2.                                            **
**                                                                        **
**  rsp2 is free software: you can redistribute it and/or modify it under **
**  the terms of the GNU General Public License as published by the Free  **
**  Software Foundation, either version 3 of the License, or (at your     **
**  option) any later version.                                            **
**                                                                        **
**      http://www.gnu.org/licenses/                                      **
**                                                                        **
** Do note that, while the whole of rsp2 is licensed under the GPL, many  **
** parts of it are licensed under more permissive terms.                  **
** ********************************************************************** */

use std::fmt;

/// Data describing the commands which need to be sent to lammps to initialize
/// atom types and the potential.
#[derive(Debug, Clone)]
pub struct InitInfo {
    /// Mass of each atom type.
    pub masses: Vec<f64>,

    /// Lammps commands to initialize the pair potentials.
    pub pair_style: PairStyle,

    /// Lammps commands to initialize the pair potentials.
    pub pair_coeffs: Vec<PairCoeff>,
}

/// Represents a `pair_style` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairStyle(pub Arg, pub Vec<Arg>);
/// Represents a `pair_coeff` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairCoeff(pub AtomTypeRange, pub AtomTypeRange, pub Vec<Arg>);

impl PairStyle {
    pub fn named(name: impl ToString) -> Self
    { PairStyle(Arg::from(name), vec![]) }

    pub fn name(&self) -> &str
    { &(self.0).0 }

    /// Append an argument
    pub fn arg(mut self, arg: impl ToString) -> Self
    { self.1.push(Arg::from(arg)); self }

    /// Append several uniformly-typed arguments
    pub fn args<As>(self, args: As) -> Self
        where As: IntoIterator, As::Item: ToString,
    { args.into_iter().fold(self, Self::arg) }
}

impl PairCoeff {
    pub fn new<I, J>(i: I, j: J) -> Self
        where AtomTypeRange: From<I> + From<J>,
    { PairCoeff(i.into(), j.into(), vec![]) }

    /// Append an argument
    pub fn arg(mut self, arg: impl ToString) -> Self
    { self.2.push(Arg::from(arg)); self }

    /// Append several uniformly-typed arguments
    pub fn args<As>(self, args: As) -> Self
        where As: IntoIterator, As::Item: ToString,
    { args.into_iter().fold(self, Self::arg) }
}

/// A range of AtomTypes representing the star-wildcard ranges
/// accepted by the `pair_coeff` command.
///
/// Construct like `typ.into()` or `(..).into()`.
//
// (NOTE: This is stored as the doubly-inclusive range sent
//        to Lammps. We store ints instead of AtomTypes so that
//        it can represent the empty range "1*0", but I haven't
//        tested whether LAMMPS actually even allows that)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomTypeRange(Option<i64>, Option<i64>);

impl From<AtomType> for AtomTypeRange {
    fn from(i: AtomType) -> Self
    { AtomTypeRange(Some(i.value()), Some(i.value())) }
}

impl From<::std::ops::RangeFull> for AtomTypeRange {
    fn from(_: ::std::ops::RangeFull) -> Self
    { AtomTypeRange(None, None) }
}

impl From<::std::ops::Range<AtomType>> for AtomTypeRange {
    fn from(r: ::std::ops::Range<AtomType>) -> Self {
        // (adjust because we take half-inclusive, but store doubly-inclusive)
        AtomTypeRange(Some(r.start.value()), Some(r.end.value() - 1))
    }
}

impl fmt::Display for AtomTypeRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn write_endpoint(f: &mut fmt::Formatter<'_>, i: Option<i64>) -> fmt::Result {
            match i {
                Some(i) => write!(f, "{}", i),
                None => Ok(()),
            }
        }
        let AtomTypeRange(min, max) = *self;
        write_endpoint(f, min)?;
        write!(f, "*")?;
        write_endpoint(f, max)?;
        Ok(())
    }
}

//-------------------------------------------

/// Type used for stringy arguments to a Lammps command,
/// which takes care of quoting for interior whitespace.
///
/// (**NOTE:** actually it does not do this yet; this type is
///  simply used wherever we know quoting *should* happen
///  once implemented)
///
/// Construct using `s.into()`/`Arg::from(s)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg(pub String);

impl Arg {
    // NOTE: This isn't a From impl because the Display impl
    //       implies that Arg: ToString, and thus From<S: ToString>
    //       would conflict with the blanket From<Self> impl.
    //
    //       Save us, specialization!
    fn from<S: ToString>(s: S) -> Arg { Arg(s.to_string()) }
}

impl fmt::Display for Arg {
    // TODO: Actually handle quoting. (low priority)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    { write!(f, "{}", self.0) }
}

impl fmt::Display for PairStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {Ok({
        let PairStyle(name, args) = self;
        write!(f, "pair_style {} {}", name, ws_join(args))?;
    })}
}

impl fmt::Display for PairCoeff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {Ok({
        let PairCoeff(i, j, args) = self;
        write!(f, "pair_coeff {} {} {}", i, j, ws_join(args))?;
    })}
}

fn ws_join(items: &[Arg]) -> JoinDisplay<'_, Arg>
{ JoinDisplay { items, sep: " " } }

// Utility Display adapter for writing a separator between items.
struct JoinDisplay<'a, D> {
    items: &'a [D],
    sep: &'a str,
}

impl<'a, D: fmt::Display> fmt::Display for JoinDisplay<'a, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {Ok({
        let mut items = self.items.iter();

        if let Some(item) = items.next() {
            write!(f, "{}", item)?;
        }
        for item in items {
            write!(f, "{}{}", self.sep, item)?;
        }
    })}
}

//-------------------------------------------

pub use self::atom_type::AtomType;
// mod to encapsulate type invariant
mod atom_type {
    use super::*;

    /// A Lammps atom type.  These are numbered from 1.
    #[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
    pub struct AtomType(
        // INVARIANT: value is >= 1.
        i64,
    );

    impl AtomType {
        /// # Panics
        ///
        /// Panics on values less than 1.
        pub fn new(x: i64) -> Self {
            assert!(x > 0);
            AtomType(x as _)
        }
        pub fn value(self) -> i64 { self.0 }

        // because this is a PITA to do manually all the time...
        /// Construct from a 0-based index.
        pub fn from_index(x: usize) -> Self { AtomType((x + 1) as _) }
        /// Recover the 0-based index.
        pub fn to_index(self) -> usize { self.0 as usize - 1 }
    }

    impl fmt::Display for AtomType {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(&self.0, f)
        }
    }
}
