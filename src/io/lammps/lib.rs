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
#![allow(unused_unsafe)]
#![deny(unused_must_use)]

extern crate slice_of_array;
extern crate rsp2_structure;
extern crate rsp2_array_types;
extern crate lammps_sys;
#[macro_use] extern crate log;
#[macro_use] extern crate failure;
#[macro_use] extern crate lazy_static;
extern crate chrono;

use ::failure::Backtrace;
use ::rsp2_array_types::{V3, Unvee, Envee};
use ::log::Level;

pub type FailResult<T> = Result<T, ::failure::Error>;

use std::fmt;

pub const API_TRACE_TARGET: &'static str = concat!(module_path!(), "::c_api");
pub const API_TRACE_LEVEL: Level = Level::Trace;

/// An error thrown by the LAMMPS C API.
#[derive(Debug, Fail)]
pub struct LammpsError {
    backtrace: Backtrace,
    severity: Severity,
    message: String,
}
impl fmt::Display for LammpsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, "LAMMPS threw {}: {}",
            match self.severity {
                Severity::Recoverable => "an exception",
                Severity::Fatal => "a fatal exception",
            },
            self.message,
        )
    }
}

use ::low_level::{ComputeStyle, Skews};
pub use ::low_level::Severity;
use low_level::LammpsOwner;
mod low_level;

use ::std::path::{Path, PathBuf};
use ::slice_of_array::prelude::*;
use ::rsp2_structure::{Coords, Lattice};

pub struct Lammps<P: Potential> {
    /// Put Lammps behind a RefCell so we can paper over things like `get_natoms(&mut self)`
    /// without needing to manually verify that no mutation occurs in the Lammps source.
    ///
    /// We don't want to slam user code with runtime violations of RefCell's aliasing checks.
    /// Thankfully, `RefCell` isn't `Sync`, so we only need to worry about single-threaded
    /// violations. I don't think these will be able to occur so long as we are careful with
    /// our API:
    ///
    /// - Be careful providing methods which return a value that extends the lifetime
    ///    of an immutably-borrowed (`&self`) receiver.
    /// - Be careful providing methods that borrow `&self` and take a callback.
    ///
    /// (NOTE: I'm not entirely sure if this is correct.)
    ptr: ::std::cell::RefCell<LammpsOwner>,

    /// This is stored to help convert metadata to/from AtomTypes.
    potential: P,

    /// The currently computed structure, encapsulated in a helper type
    /// that tracks dirtiness and helps us decide when we need to call lammps
    structure: MaybeDirty<(Coords, P::Meta)>,

    auto_adjust_lattice: bool,

    // These store data about the structure given to Builder::build
    // which must remain constant in all calls to `compute_*`.
    // See the documentation of `build` for more info.
    original_num_atoms: usize,
    original_init_info: InitInfo,

    // Determines the next command for updating.
    update_fsm: UpdateFsm,

    data_trace_dir: Option<PathBuf>,
}

struct MaybeDirty<T> {
    // NOTE: Possible states for the members:
    //
    //        dirty:       clean:       when
    //        Some(s)       None       is dirty, and has never been clean.
    //        Some(s)      Some(s)     is dirty, but has been clean in the past.
    //         None        Some(s)     is currently clean.

    /// new data that has not been marked clean.
    dirty: Option<T>,
    /// the last data marked clean.
    clean: Option<T>,
}

impl<T> MaybeDirty<T> {
    pub fn new_dirty(x: T) -> MaybeDirty<T> {
        MaybeDirty {
            dirty: Some(x),
            clean: None,
        }
    }

    pub fn is_dirty(&self) -> bool
    { self.dirty.is_some() }

    pub fn last_clean(&self) -> Option<&T>
    { self.clean.as_ref() }

    pub fn get(&self) -> &T
    { self.dirty.as_ref().or(self.last_clean()).unwrap() }

    /// Get a mutable reference. This automatically marks the value as dirty.
    pub fn get_mut(&mut self) -> &mut T
    where T: Clone,
    {
        if self.dirty.is_none() {
            self.dirty = self.clean.clone();
        }
        self.dirty.as_mut().unwrap()
    }

    pub fn mark_clean(&mut self)
    {
        assert!(self.dirty.is_some() || self.clean.is_some());

        if self.dirty.is_some() {
            self.clean = self.dirty.take();
        }

        assert!(self.dirty.is_none());
        assert!(self.clean.is_some());
    }

    // test if f(x) is dirty by equality
    // HACK
    // this is only provided to help work around borrow checker issues
    //
    // To clarify: If there is no last clean value, then ALL projections
    // are considered dirty by definition.
    pub fn is_projection_dirty<K: ?Sized + PartialEq>(
        &self,
        mut f: impl FnMut(&T) -> &K,
    ) -> bool {
        match (&self.clean, &self.dirty) {
            (Some(clean), Some(dirty)) => f(clean) != f(dirty),
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => unreachable!(),
        }
    }

    // HACK
    // This differs from `is_projection_dirty` only in that the callback
    // returns owned data instead of borrowed. One might think that this
    // method could therefore be used to implement the other; but it can't,
    // because the lifetime in F's return type would be overconstrained.
    pub fn is_function_dirty<K: PartialEq>(
        &self,
        mut f: impl FnMut(&T) -> K,
    ) -> bool {
        match (&self.clean, &self.dirty) {
            (Some(clean), Some(dirty)) => f(clean) != f(dirty),
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Builder {
    append_log: Option<PathBuf>,
    threaded: bool,
    auto_adjust_lattice: bool,
    update_style: UpdateStyle,
    data_trace_dir: Option<PathBuf>,
}

//------------------------------------------

/// Configuration for how to tell LAMMPS to update.
///
/// `run $n`. `run 0` is a safe way to do a full update.
#[derive(Debug, Clone)]
pub struct UpdateStyle {
    /// The N in `run N pre PRE post POST`
    pub n: u32,
    /// `pre yes` or `pre no`. Notice Lammps overrides this on the first call
    /// to always be true.
    pub pre: bool,
    pub post: bool,
    /// Send exact positions at this interval. (`0` = never)
    ///
    /// On other steps, read back the previous positions from Lammps and add the relative
    /// change.  This is less likely to trigger neighbor list updates for `pre no`, but
    /// rounding errors will lead to an accumulation of numerical discrepancies between the
    /// input structure and the one seen by lammps.
    pub sync_positions_every: u32,
}

impl UpdateStyle {
    pub fn safe() -> Self
    { UpdateStyle { n: 0, pre: true, post: true, sync_positions_every: 1 } }

    pub fn fast(sync_positions_every: u32) -> Self
    { UpdateStyle { n: 1, pre: false, post: false, sync_positions_every } }
}

// Determines the next `run` command for updating Lammps.
//
// Now that we always use the same command, the implementation is trivial.
// It only exists as a holdover from an earlier, less trivial design.
#[derive(Debug, Clone)]
struct UpdateFsm {
    iter: u32,
    style: UpdateStyle,
}

#[derive(Debug, Clone)]
struct UpdateAction {
    command: String,
    positions: UpdatePositions,
}

#[derive(Debug, Copy, Clone)]
enum UpdatePositions { Relative, Absolute }

impl UpdateStyle {
    fn initial_fsm(&self) -> UpdateFsm {
        UpdateFsm { iter: 0, style: self.clone() }
    }
}

impl UpdateFsm {
    fn step(&mut self) -> UpdateAction {
        let action = self._action();
        self.iter += 1;
        action
    }

    fn _action(&self) -> UpdateAction {
        struct YesNo(bool);
        impl fmt::Display for YesNo {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", if self.0 { "yes" } else { "no" })
            }
        }

        let UpdateStyle { n, pre, post, sync_positions_every } = self.style;
        let positions = match (self.iter, sync_positions_every) {
            (0, 0) => UpdatePositions::Absolute,
            (_, 0) => UpdatePositions::Relative,
            (i, m) => match i % m {
                0 => UpdatePositions::Absolute,
                _ => UpdatePositions::Relative,
            },
        };
        let command = format!("run {} pre {} post {}", n, YesNo(pre), YesNo(post));

        UpdateAction { positions, command }
    }
}

//------------------------------------------

impl Default for Builder {
    fn default() -> Self
    { Builder::new() }
}

impl Builder {
    pub fn new() -> Self
    { Builder {
        append_log: None,
        threaded: true,
        update_style: UpdateStyle::safe(),
        auto_adjust_lattice: true,
        data_trace_dir: None,
    }}

    /// Toggles extremely small corrections automatically made to the lattice.
    ///
    /// By default, corrections of up to a single ULP per lattice element (i.e. next
    /// or previous representable float) will be made to the lattice before it is
    /// communicated to lammps, to help meet its draconian requirements on skew.
    /// (Lammps requires that off-diagonals do not exceed 0.5 times their diagonal
    ///  elements, and there is no fuzz to this check)
    pub fn auto_adjust_lattice(&mut self, value: bool) -> &mut Self
    { self.auto_adjust_lattice = value; self }

    pub fn append_log(&mut self, path: impl AsRef<Path>) -> &mut Self
    { self.append_log = Some(path.as_ref().to_owned()); self }

    pub fn threaded(&mut self, value: bool) -> &mut Self
    { self.threaded = value; self }

    pub fn update_style(&mut self, value: UpdateStyle) -> &mut Self
    { self.update_style = value; self }

    pub fn data_trace_dir(&mut self, value: Option<impl AsRef<Path>>) -> &mut Self
    { self.data_trace_dir = value.map(|p| p.as_ref().to_owned()); self }

    /// Call out to the LAMMPS C API to create an instance of Lammps,
    /// and configure it according to this builder.
    ///
    /// # The "initial structure" arguments
    ///
    /// It may seem unusual that both `build` and the `Lammps::compute_*` methods
    /// require a structure.  The reason is because certain API calls made during
    /// initialization depend on certain properties of the structure.
    ///
    /// To simplify the implementation of `Lammps`, the following properties
    /// are effectively **set in stone** after this method:
    ///
    /// * The number of atoms
    /// * The masses of each atom type
    /// * The sequence of `pair_style` and `pair_coeff` commands needed
    ///   to initialize the potential
    ///
    /// The `compute_*` methods on `Lammps` will check these properties on every
    /// computed structure, and will fail if they disagree with the structure
    /// that was initially provided to `build`.
    pub fn build<P>(&self, potential: P, initial_coords: Coords, initial_meta: P::Meta) -> FailResult<Lammps<P>>
    where P: Potential,
    { Lammps::from_builder(self, potential, initial_coords, initial_meta) }

    /// Create a `DispFn`, an alternative to `Lammps` which is optimized for computing
    /// forces at displacements.
    pub fn build_disp_fn<P>(&self, potential: P, equilibrium_coords: Coords, equilibrium_meta: P::Meta) -> FailResult<DispFn<P>>
    where P: Potential,
    { DispFn::from_builder(self, potential, equilibrium_coords, equilibrium_meta) }
}

/// Initialize LAMMPS, do nothing of particular value, and exit.
///
/// For debugging linker errors.
pub fn link_test() -> FailResult<()>
{Ok({
    let _ = ::LammpsOwner::new(&["lammps",
        "-screen", "none",
        "-log", "none",
    ])?;
})}

pub use atom_type::AtomType;
// mod to encapsulate type invariant
mod atom_type {
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
}

/// Trait through which a consumer of this crate sets up the potential to be used.
///
/// It is used to initialize the atom types and pair potentials when Lammps is initialized.
///
/// The `Meta` associated type will be the metadata type accepted by e.g. `set_structure`.
/// Feel free to pick something convenient for your application.
pub trait Potential {

    type Meta: Clone;

    /// Produce information needed by `rsp2_lammps_wrap` to initialize the potential.
    ///
    /// See `InitInfo` for more information.
    ///
    /// Currently, this crate does not support modification of the defined
    /// atom types or the potential on an existing Lammps.
    /// This method will be called once on the structure used to build a Lammps
    /// (to generate the initialization commands), as well as before each computation
    /// (to verify that the necessary commands have not changed).
    fn init_info(&self, coords: &Coords, meta: &Self::Meta) -> InitInfo;

    /// Assign atom types to each atom.
    ///
    /// The reason this responsibility is deferred to code outside this crate is
    /// because different potentials may need to use atom types in wildly different
    /// ways. For many potentials, the atom types will simply map to elemental
    /// species; but for a potential like 'kolmogorov/crespi/z', atom types must be
    /// carefully assigned to prevent interactions between non-adjacent layers.
    fn atom_types(&self, coords: &Coords, meta: &Self::Meta) -> Vec<AtomType>;
}

impl<'a, M: Clone> Potential for Box<Potential<Meta=M> + 'a> {
    type Meta = M;

    fn init_info(&self, coords: &Coords, meta: &Self::Meta) -> InitInfo
    { (&**self).init_info(coords, meta) }

    fn atom_types(&self, coords: &Coords, meta: &Self::Meta) -> Vec<AtomType>
    { (&**self).atom_types(coords, meta) }
}

impl<'a, M: Clone> Potential for &'a (Potential<Meta=M> + 'a) {
    type Meta = M;

    fn init_info(&self, coords: &Coords, meta: &Self::Meta) -> InitInfo
    { (&**self).init_info(coords, meta) }

    fn atom_types(&self, coords: &Coords, meta: &Self::Meta) -> Vec<AtomType>
    { (&**self).atom_types(coords, meta) }
}

//-------------------------------------------

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
    fn from(r: ::std::ops::Range<AtomType>) -> Self
    {
        // (adjust because we take half-inclusive, but store doubly-inclusive)
        AtomTypeRange(Some(r.start.value()), Some(r.end.value() - 1))
    }
}

impl fmt::Display for AtomTypeRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn write_endpoint(f: &mut fmt::Formatter, i: Option<i64>) -> fmt::Result {
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    { write!(f, "{}", self.0) }
}

impl fmt::Display for PairStyle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {Ok({
        let PairStyle(name, args) = self;
        write!(f, "pair_style {} {}", name, ws_join(args))?;
    })}
}

impl fmt::Display for PairCoeff {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {Ok({
        let PairCoeff(i, j, args) = self;
        write!(f, "pair_coeff {} {} {}", i, j, ws_join(args))?;
    })}
}

fn ws_join(items: &[Arg]) -> JoinDisplay<'_, Arg> {
    JoinDisplay { items, sep: " " }
}

// Utility Display adapter for writing a separator between items.
struct JoinDisplay<'a, D: 'a> {
    items: &'a [D],
    sep: &'a str,
}

impl<'a, D: fmt::Display> fmt::Display for JoinDisplay<'a, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
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
// Initializing the LAMMPS C API object.
//
impl<P: Potential> Lammps<P>
{
    // implementation of Builder::Build
    fn from_builder(builder: &Builder, potential: P, coords: Coords, meta: P::Meta) -> FailResult<Self>
    {Ok({
        let original_num_atoms = coords.num_atoms();
        let original_init_info = potential.init_info(&coords, &meta);

        let ptr = Self::_from_builder(builder, original_num_atoms, &original_init_info)?;
        Lammps {
            ptr: ::std::cell::RefCell::new(ptr),
            structure: MaybeDirty::new_dirty((coords, meta)),
            potential,
            original_init_info,
            original_num_atoms,
            auto_adjust_lattice: builder.auto_adjust_lattice,
            update_fsm: builder.update_style.initial_fsm(),
            data_trace_dir: builder.data_trace_dir.clone(),
        }
    })}

    // monomorphic
    fn _from_builder(
        builder: &Builder,
        num_atoms: usize,
        init_info: &InitInfo,
    ) -> FailResult<LammpsOwner>
    {Ok({
        // Lammps script based on code from Colin Daniels.

        let mut lmp = ::LammpsOwner::new(&[
            "lammps",
            "-screen", "none",
            "-log", "none", // logs opened from CLI are truncated, but we want to append
        ])?;

        if let Some(log_file) = &builder.append_log {
            // Append a header to the log file as a feeble attempt to help delimit individual
            // runs (even though it will still get messy for parallel runs).
            //
            // NOTE: This looks like a surprising hidden side-effect, but it isn't really.
            //       Or rather, that is to say, removing it won't make things any better,
            //       because LAMMPS itself will be writing many things to this same file
            //       anyways over the course of this function.
            //
            // Errs are ignored because *it's three lines in a stinking log file*.
            use ::std::io::prelude::*;
            if let Ok(mut f) =
                ::std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .append(true)
                    .open(log_file)
            {
                let _ = writeln!(f, "---------------------------------------------");
                let _ = writeln!(f, "---- Begin run at {}", ::chrono::Local::now());
                let _ = writeln!(f, "---------------------------------------------");
            }

            lmp.command(
                &format!("log {} append", log_file.display()),
            )?;
        }

        lmp.commands(&[
            "package omp 0",
            "units metal",                  // Angstroms, picoseconds, eV
            match builder.threaded {
                true => "processors * * *",
                false => "processors 1 1 1",
            },
            "neigh_modify delay 0",         // disable delay for a safer `run pre no`
            "atom_style atomic",            // attributes to store per-atom
            "thermo_modify lost error",     // don't let atoms disappear without telling us
            "atom_modify map array",        // store all positions in an array
            "atom_modify sort 0 0.0",       // don't reorder atoms during simulation
        ])?;

        // (mostly) garbage initial lattice
        lmp.commands(&[
            "boundary p p p",               // (p)eriodic, (f)ixed, (s)hrinkwrap
            "box tilt small",               // triclinic
            // NOTE: Initial skew factors must be zero to simplify
            //       reasoning about order in send_lmp_lattice.
            "region sim prism 0 2 0 2 0 2 0 0 0", // garbage garbage garbage
        ])?;

        {
            let InitInfo { masses, pair_style, pair_coeffs } = init_info;

            lmp.command(&format!("create_box {} sim", masses.len()))?;
            for (i, mass) in (1..).zip(masses) {
                lmp.command(&format!("mass {} {}", i, mass))?;
            }

            lmp.command(pair_style)?;
            lmp.commands(pair_coeffs)?;
        }

        // garbage initial positions
        {
            let this_atom_type = 1;
            let seed = 0xbeef;
            lmp.command(
                &format!("create_atoms {} random {} {} NULL remap yes",
                this_atom_type, num_atoms, seed))?;
        }

        // set up computes
        lmp.commands(&[
            &format!("compute RSP2_PE all pe"),
            &format!("compute RSP2_Pressure all pressure NULL virial"),
        ])?;

        lmp
    })}
}

//-------------------------------------------
// Public API for modifying the system to be computed.
//
// All these need to do is modify the Structure object through the dirtying API.
//
// There's nothing tricky that needs to be done here to ensure that updates are
// propagated to LAMMPS, so long as `update_computation` is able to correctly
// detect when the new values differ from the old.
impl<P: Potential> Lammps<P> {
    pub fn set_structure(&mut self, new: Coords, meta: P::Meta) -> FailResult<()>
    {Ok({
        *self.structure.get_mut() = (new, meta);
    })}

    pub fn set_carts(&mut self, new: &[V3]) -> FailResult<()>
    {Ok({
        self.structure.get_mut().0.set_carts(new.to_vec());
    })}

    pub fn set_lattice(&mut self, new: Lattice) -> FailResult<()>
    {Ok({
        self.structure.get_mut().0.set_lattice(&new);
    })}
}

//-------------------------------------------
// sending input to lammps and running the main computation

impl<P: Potential> Lammps<P> {
    // This will rerun computations in lammps, but only if things have changed.
    // (The very first time it is called is also when the everything in LAMMPS
    //  will be populated with initial data that isn't just garbage.)
    //
    // At the end, (cached, updated) == (Some(_), None)
    fn update_computation(&mut self) -> FailResult<()>
    {Ok({
        if self.structure.is_dirty() {
            self.structure.get_mut().0.ensure_carts();

            {
                let (coords, meta) = self.structure.get();
                self.check_data_set_in_stone(coords, meta)?;
            }

            let iter = self.update_fsm.iter;
            let UpdateAction { command, positions: update_positions } = self.update_fsm.step();

            // Only send data that has changed from the cache.
            // This is done because it appears that lammps does some form of
            // caching as well (lattice modifications in particular appear
            // to increase the amount of computational work)
            //
            // The first time through, all projections will be considered dirty,
            // so everything will be sent to replace the garbage data that we
            // gave lammps during initialization.

            // NOTE: we end up calling Potential::atom_types() more often than
            //       I would have liked (considering that it might e.g. run an
            //       algorithm to assign layer numbers to all atoms), but w/e.
            if self.structure.is_function_dirty(|&(ref c, ref m)| self.potential.atom_types(c, m)) {
                self.send_lmp_types()?;
            }

            if self.structure.is_projection_dirty(|&(ref c, _)| c.lattice()) {
                self.send_lmp_lattice()?;
            }

            if self.structure.is_projection_dirty(|&(ref c, _)| c.as_carts_cached().unwrap()) {
                self.send_lmp_carts(update_positions)?;
            }

            self.update_lammps_with_command(iter, &command)?;
            self.structure.mark_clean();
        }
    })}

    fn update_lammps_with_command(&mut self, iter: u32, command: &str) -> FailResult<()>
    {Ok({
        if let Some(dir) = &self.data_trace_dir {
            self.write_data_trace_fileset(dir, &format!("{:04}-a", iter));
        }

        self.ptr.borrow_mut().command(command)?;

        if let Some(dir) = &self.data_trace_dir {
            self.write_data_trace_fileset(dir, &format!("{:04}-b", iter));
        }
    })}

    fn send_lmp_types(&mut self) -> FailResult<()>
    {Ok({
        let types = {
            let (coords, meta) = self.structure.get();
            self.potential.atom_types(coords, meta)
        };
        assert_eq!(types.len(), self.ptr.borrow_mut().get_natoms());

        let types = types.into_iter().map(AtomType::value).collect::<Vec<_>>();

        unsafe { self.ptr.borrow_mut().scatter_atoms_i("type", &types) }?;
    })}

    fn send_lmp_carts(&mut self, style: UpdatePositions) -> FailResult<()>
    {Ok({
        let new_user_carts = self.structure.get().0.as_carts_cached().expect("(BUG)");

        let new_lmp_carts = match style {
            UpdatePositions::Absolute => new_user_carts.to_vec(),
            UpdatePositions::Relative => {
                let old_user_coords = &self.structure.last_clean().expect("(BUG) first step can't be relative").0;
                let old_user_carts = old_user_coords.as_carts_cached().expect("(BUG)");
                let mut lmp_carts = self.read_raw_lmp_carts()?;

                { // scope mut borrow.  NLL, come save us!
                    let iter = old_user_carts.iter().zip(new_user_carts).zip(&mut lmp_carts);
                    for ((old_user, new_user), lmp) in iter {
                        *lmp += new_user - old_user;
                    }
                }
                lmp_carts
            }
        };
        unsafe { self.ptr.borrow_mut().scatter_atoms_f("x", new_lmp_carts.unvee_ref().flat()) }?;
    })}

    fn send_lmp_lattice(&mut self) -> FailResult<()>
    {Ok({
        let [
            [xx, _0, _1],
            [xy, yy, _2],
            [xz, yz, zz],
        ] = self.structure.get().0.lattice().matrix().unvee();

        assert_eq!(0f64, _0, "non-triangular lattices not yet supported");
        assert_eq!(0f64, _1, "non-triangular lattices not yet supported");
        assert_eq!(0f64, _2, "non-triangular lattices not yet supported");

        let mut diag = [xx, yy, zz];
        let mut skews = Skews { xy, yz, xz };
        if self.auto_adjust_lattice {
            auto_adjust_lattice(&mut diag, &mut skews);
        }

        // safe because we initialized the lattice the hard way during initialization,
        // so surely `domain->set_initial_box()` must have been called...
        unsafe { self.ptr.borrow_mut().reset_box([0.0, 0.0, 0.0], diag, skews)? }

        // "Is this box to your liking, sire?"
        self.ptr.borrow_mut().commands(&[
            // These should have no effect, but they give LAMMPS a chance to look at the
            // box and throw an exception if it doesn't like what it sees.
            format!("change_box all x final 0 {}", diag[0]),
            format!("change_box all xy final {}", skews.xy),
        ])?;
    })}

    // Some of these properties probably could be allowed to change,
    // but it's not important enough for me to look into them right now,
    // so we simply check that they don't change.
    fn check_data_set_in_stone(&self, coords: &Coords, meta: &P::Meta) -> FailResult<()>
    {Ok({
        fn check<D>(msg: &'static str, was: D, now: D) -> FailResult<()>
        where D: fmt::Debug + PartialEq,
        {
            ensure!(
                was == now,
                format!("{} changed since build()\n was: {:?}\n now: {:?}", msg, was, now));
            Ok(())
        }
        let InitInfo { masses, pair_style, pair_coeffs } = self.potential.init_info(coords, meta);
        let Lammps {
            original_num_atoms,
            original_init_info: InitInfo {
                masses: ref original_masses,
                pair_style: ref original_pair_style,
                pair_coeffs: ref original_pair_coeffs,
            },
            ..
        } = *self;

        check("number of atom types has", original_masses.len(), masses.len())?;
        check("number of atoms has", original_num_atoms, coords.num_atoms())?;
        check("masses have", original_masses, &masses)?;
        check("pair_style command has", original_pair_style, &pair_style)?;
        check("pair_coeff commands have", original_pair_coeffs, &pair_coeffs)?;
    })}

    fn write_data_trace_fileset(&self, dir: &Path, filename_prefix: &str) {
        let _ = self._write_data_trace_fileset(dir, filename_prefix);
    }

    fn _write_data_trace_fileset(&self, dir: &Path, filename_prefix: &str) -> FailResult<()>
    {Ok({
        use ::std::fs;
        use ::std::io::Write;

        fs::create_dir_all(dir)?;
        let file = |ext: &str| {
            fs::File::create(dir.join(format!("{}.{}", filename_prefix, ext)))
        };
        writeln!(file("lmp.carts")?, "{:?}", self.read_raw_lmp_carts()?)?;
        writeln!(file("lmp.force")?, "{:?}", self.read_raw_lmp_force()?)?;
        writeln!(file("cached.carts")?, "{:?}", self.structure.get().0.to_carts())?;
    })}
}

fn auto_adjust_lattice(diag: &mut [f64; 3], skews: &mut Skews) {
    fn do_element(d: &mut f64, s: &mut f64) {
        if 2.0 * s.abs() > d.abs() {
            // shrink skew by 1 ULP
            *s = s.signum() * next_after(s.abs(), 0.0);
        }
        if 2.0 * s.abs() > d.abs() {
            // that wasn't enough?
            // then increase diag by 1 ULP
            *d = d.signum() * next_after(d.abs(), ::std::f64::INFINITY);
        }
    }

    // (actually, xx might change by up to two ULPs)
    do_element(&mut diag[0], &mut skews.xy);
    do_element(&mut diag[0], &mut skews.xz);
    do_element(&mut diag[1], &mut skews.yz);
}

use ::std::os::raw::c_double;
#[link_name = "m"]
extern {
    fn nextafter(from: c_double, to: c_double) -> c_double;
}
fn next_after(from: f64, to: f64) -> f64 {
    unsafe { nextafter(from, to) }
}

//-------------------------------------------
// direct reading of lammps' internal state, without side-effects (I hope)

impl<P: Potential> Lammps<P> {
    fn read_raw_lmp_carts(&self) -> FailResult<Vec<V3>>
    {Ok({
        let x = unsafe { self.ptr.borrow_mut().gather_atoms_f("x", 3)? };
        x.nest::<[_; 3]>().to_vec().envee()
    })}

    fn read_raw_lmp_force(&self) -> FailResult<Vec<V3>>
    {Ok({
        let x = unsafe { self.ptr.borrow_mut().gather_atoms_f("f", 3)? };
        x.nest::<[_; 3]>().to_vec().envee()
    })}
}

//-------------------------------------------
// gathering output from lammps
//
// NOTE: Every method here should call update_computation().
//       Don't worry about being sloppy with redundant calls;
//       the method was designed for such usage.
impl<P: Potential> Lammps<P> {
    /// Get the potential, possibly performing some computations if necessary.
    pub fn compute_value(&mut self) -> FailResult<f64>
    {Ok({
        self.update_computation()?;

        unsafe { self.ptr.borrow_mut().extract_compute_0d("RSP2_PE") }?
    })}

    /// Get the forces, possibly performing some computations if necessary.
    pub fn compute_force(&mut self) -> FailResult<Vec<V3>>
    {Ok({
        self.update_computation()?;
        self.read_raw_lmp_force()?
    })}

    /// Get the gradient, possibly performing some computations if necessary.
    pub fn compute_grad(&mut self) -> FailResult<Vec<V3>>
    {Ok({
        self.update_computation()?;

        let mut grad = self.compute_force()?;
        for v in &mut grad {
            *v *= -1.0;
        }
        grad
    })}

    /// Get the pressure tensor, possibly performing some computations if necessary.
    pub fn compute_pressure(&mut self) -> FailResult<[f64; 6]>
    {Ok({
        self.update_computation()?;

        unsafe {
            self.ptr.borrow_mut().extract_compute_1d("RSP2_Pressure", ComputeStyle::Global, 6)
        }?.to_array()
    })}
}

/// Pre-packaged potentials.
pub mod potential {
    use super::*;

    /// Represents 'pair_style none'.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
    pub struct None;

    impl Potential for None {
        type Meta = ();

        fn atom_types(&self, coords: &Coords, (): &()) -> Vec<AtomType>
        { vec![AtomType::new(1); coords.num_atoms()]}

        fn init_info(&self, _: &Coords, (): &()) -> InitInfo
        { InitInfo {
            masses: vec![1.0],
            pair_style: PairStyle::named("none"),
            pair_coeffs: vec![],
        }}
    }
}

//-------------------------------------------

/// An alternative to `Lammps` which is optimized for computing forces at displacements.
///
/// (Long story short, it was easier to write a wrapper type that handles the
/// coordinates correctly, rather than to try and build this functionality into `Lammps`)
pub struct DispFn<P: Potential> {
    lammps: Lammps<P>,
    // the carts produced by lammps after first building the neighbor list.
    //
    // These might be slightly different from the original coords (namely, they may be
    // mapped into the unit cell, and the mapping process will also cause sites within
    // the cell to change by a ULP or two).
    //
    // We use these as the original coords (instead of the original user input) to avoid
    // invalidating the data structures of lammps, which may rely on these particular images
    // of the atoms being chosen.
    equilibrium_carts: Vec<V3>,
    equilibrium_force: Vec<V3>,
}

impl<P: Potential> DispFn<P> {
    fn from_builder(builder: &Builder, potential: P, coords: Coords, meta: P::Meta) -> FailResult<Self>
    {Ok({
        let mut builder = builder.clone();
        builder.update_style(UpdateStyle { n: 1, pre: false, post: false, sync_positions_every: 1 });

        let mut lammps = Lammps::from_builder(&builder, potential, coords, meta)?;

        // this will build neighbor lists (modifying the coordinates in the process)
        let equilibrium_force = lammps.compute_force()?;

        let equilibrium_carts = lammps.read_raw_lmp_carts()?;

        DispFn { lammps, equilibrium_carts, equilibrium_force }
    })}

    pub fn compute_force_at_disp(&mut self, disp: (usize, V3)) -> FailResult<Vec<V3>>
    {Ok({
        let mut carts = self.equilibrium_carts.clone();
        carts[disp.0] += disp.1;
        self.lammps.set_carts(&carts)?;
        self.lammps.compute_force()?
    })}

    pub fn equilibrium_force(&self) -> Vec<V3>
    { self.equilibrium_force.clone() }
}

//-------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: this is now mostly tested indirectly
    //       through the other crates that use it.

    // get a fresh Lammps instance on which arbitrary functions can be called.
    fn arbitrary_initialized_lammps() -> Lammps<potential::None>
    {
        use ::rsp2_structure::CoordsKind;
        let coords = Coords::new(
            Lattice::eye(),
            CoordsKind::Fracs(vec![V3([0.0; 3])]),
        );
        Builder::new().build(Default::default(), coords, ()).unwrap()
    }

    macro_rules! assert_matches {
        ($pat:pat $(if $cond:expr)*, $val:expr $(,)*)
        => {
            match $val {
                $pat $(if $cond)* => {},
                ref e => {
                    panic!(
                        "assert_matches! failed:\nExpected:{}\n  Actual:{:?}",
                        stringify!($pat), e);
                },
            }
        };
    }

    #[test]
    fn exceptions()
    {
        let lmp = arbitrary_initialized_lammps();
        let e = unsafe { lmp.ptr.borrow_mut().commands(&[
            // try to change to block with a nonzero skew
            "change_box all xy final 0.25",
            "change_box all ortho",
        ]).unwrap_err() };

        assert_matches!(
            LammpsError {
                severity: Severity::Recoverable,
                ..
            },
            e.downcast().expect("wrong error type"),
        );
    }
}

#[cfg(test)]
mod compiletest {
    use super::*;

    fn assert_send<S: Send>() {}
    fn assert_sync<S: Sync>() {}

    #[test]
    fn builder_is_send_and_sync() {
        assert_send::<Builder>();
        assert_sync::<Builder>();
    }
}
