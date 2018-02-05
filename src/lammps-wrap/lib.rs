/* FIXME need GNU GPL header
 */

#![allow(unused_unsafe)]

extern crate slice_of_array;
extern crate rsp2_structure;
extern crate lammps_sys;
#[macro_use] extern crate log;
#[macro_use] extern crate error_chain;
#[macro_use] extern crate lazy_static;
extern crate chrono;

use std::fmt;

error_chain! {
    foreign_links {
        NulError(::std::ffi::NulError);
    }

    errors {
        Lammps(severity: Severity, message: String) {
            description("LAMMPS threw an exception"),
            display("LAMMPS threw {}: {}",
                match *severity {
                    Severity::Recoverable => "an exception",
                    Severity::Fatal => "a fatal exception",
                },
                message
            ),
        }
        BadMeta(potential: &'static str, value_debug: String) {
            description("Bad atom metadata for potential"),
            display("Bad atom metadata for potential {}: {}", potential, value_debug),
        }
    }
}

// constructs an Error, as opposed to bail!() which diverges with a Result.
// Useful in `ok_or_else`.
macro_rules! err {
    ($($t:tt)+)
    => { Error::from(format!($($t)+)) }
}

pub type StdResult<T, E> = ::std::result::Result<T, E>;

use ::low_level::ComputeStyle;
pub use ::low_level::Severity;
use low_level::LammpsOwner;
mod low_level;

use ::std::path::{Path, PathBuf};
use ::slice_of_array::prelude::*;
use ::rsp2_structure::{Structure, Lattice};

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
    structure: MaybeDirty<Structure<P::Meta>>,

    // These store data about the structure given to Builder::build
    // which must remain constant in all calls to `compute_*`.
    // See the documentation of `build` for more info.
    original_num_atoms: usize,
    original_init_info: InitInfo,
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
    pub fn is_projection_dirty<K: ?Sized, F>(&self, mut f: F) -> bool
    where
        K: PartialEq,
        F: FnMut(&T) -> &K,
    {
        match (&self.clean, &self.dirty) {
            (&Some(ref clean), &Some(ref dirty)) => f(clean) != f(dirty),
            (&None, &Some(_)) => true,
            (&Some(_), &None) => false,
            (&None, &None) => unreachable!(),
        }
    }

    // HACK
    // This differs from `is_projection_dirty` only in that the callback
    // returns owned data instead of borrowed. One might think that this
    // method could therefore be used to implement the other; but it can't,
    // because the lifetime in F's return type would be overconstrained.
    pub fn is_function_dirty<K, F>(&self, mut f: F) -> bool
    where
        K: PartialEq,
        F: FnMut(&T) -> K,
    {
        match (&self.clean, &self.dirty) {
            (&Some(ref clean), &Some(ref dirty)) => f(clean) != f(dirty),
            (&None, &Some(_)) => true,
            (&Some(_), &None) => false,
            (&None, &None) => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Builder {
    append_log: Option<PathBuf>,
    threaded: bool,
}

impl Default for Builder {
    fn default() -> Self
    { Builder::new() }
}

impl Builder {
    pub fn new() -> Self
    { Builder {
        append_log: None,
        threaded: true,
    }}

    pub fn append_log<P: AsRef<Path>>(&mut self, path: P) -> &mut Self
    { self.append_log = Some(path.as_ref().to_owned()); self }

    pub fn threaded(&mut self, value: bool) -> &mut Self
    { self.threaded = value; self }

    /// Call out to the LAMMPS C API to create an instance of Lammps,
    /// and configure it according to this builder.
    ///
    /// # The `initial_structure` argument
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
    pub fn build<P>(&self, potential: P, initial_structure: Structure<P::Meta>) -> Result<Lammps<P>>
    where P: Potential,
    { Lammps::from_builder(self, potential, initial_structure) }
}

/// Initialize LAMMPS, do nothing of particular value, and exit.
///
/// For debugging linker errors.
pub fn link_test() -> Result<()>
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
    fn init_info(&self, structure: &Structure<Self::Meta>) -> InitInfo;

    /// Assign atom types to each atom.
    ///
    /// The reason this responsibility is deferred to code outside this crate is
    /// because different potentials may need to use atom types in wildly different
    /// ways. For many potentials, the atom types will simply map to elemental
    /// species; but for a potential like 'kolmogorov/crespi/z', atom types must be
    /// carefully assigned to prevent interactions between non-adjacent layers.
    fn atom_types(&self, structure: &Structure<Self::Meta>) -> Vec<AtomType>;
}

impl<'a, M: Clone> Potential for Box<Potential<Meta=M> + 'a> {
    type Meta = M;

    fn init_info(&self, structure: &Structure<Self::Meta>) -> InitInfo
    { (&**self).init_info(structure) }

    fn atom_types(&self, structure: &Structure<Self::Meta>) -> Vec<AtomType>
    { (&**self).atom_types(structure) }
}

impl<'a, M: Clone> Potential for &'a (Potential<Meta=M> + 'a) {
    type Meta = M;

    fn init_info(&self, structure: &Structure<Self::Meta>) -> InitInfo
    { (&**self).init_info(structure) }

    fn atom_types(&self, structure: &Structure<Self::Meta>) -> Vec<AtomType>
    { (&**self).atom_types(structure) }
}

/// Data describing the commands which need to be sent to lammps to initialize
/// atom types and the potential.
#[derive(Debug, Clone)]
pub struct InitInfo {
    /// Mass of each atom type.
    pub masses: Vec<f64>,

    /// Lammps commands to initialize the pair potentials.
    pub pair_commands: Vec<PairCommand>,
}

/// Represents commands allowed to appear in `InitInfo.pair_commands`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairCommand {
    PairStyle(Arg, Vec<Arg>),
    PairCoeff(AtomTypeRange, AtomTypeRange, Vec<Arg>),
}

impl PairCommand {
    pub fn pair_style<S>(name: S) -> Self
    where S: ToString,
    { PairCommand::PairStyle(Arg::from(name), vec![]) }

    pub fn pair_coeff<I, J>(i: I, j: J) -> Self
    where AtomTypeRange: From<I> + From<J>,
    { PairCommand::PairCoeff(i.into(), j.into(), vec![]) }

    /// Append an argument
    pub fn arg<A>(mut self, arg: A) -> Self
    where A: ToString,
    {
        match self {
            PairCommand::PairCoeff(_, _, ref mut v) => v,
            PairCommand::PairStyle(_, ref mut v) => v,
        }.push(Arg::from(arg));
        self
    }

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

impl fmt::Display for PairCommand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {Ok({
        fn ws_join(items: &[Arg]) -> JoinDisplay<Arg> {
            JoinDisplay { items, sep: " " }
        }

        match *self {
            PairCommand::PairStyle(ref name, ref args) => {
                write!(f, "pair_style {} {}", name, ws_join(args))?;
            },
            PairCommand::PairCoeff(ref i, ref j, ref args) => {
                write!(f, "pair_coeff {} {} {}", i, j, ws_join(args))?;
            },
        }
    })}
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
    fn from_builder(builder: &Builder, potential: P, structure: Structure<P::Meta>) -> Result<Self>
    {Ok({
        let original_num_atoms = structure.num_atoms();
        let original_init_info = potential.init_info(&structure);

        let ptr = Self::_from_builder(builder, original_num_atoms, &original_init_info)?;
        Lammps {
            ptr: ::std::cell::RefCell::new(ptr),
            structure: MaybeDirty::new_dirty(structure),
            potential,
            original_init_info,
            original_num_atoms,
        }
    })}

    // monomorphic
    fn _from_builder(
        builder: &Builder,
        num_atoms: usize,
        init_info: &InitInfo,
    ) -> Result<LammpsOwner>
    {Ok({
        // Lammps script based on code from Colin Daniels.

        let mut lmp = ::LammpsOwner::new(&[
            "lammps",
            "-screen", "none",
            "-log", "none", // logs opened from CLI are truncated, but we want to append
        ])?;

        if let Some(ref log_file) = builder.append_log
        {
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
            let InitInfo { ref masses, ref pair_commands } = *init_info;

            lmp.command(&format!("create_box {} sim", masses.len()))?;
            for (i, mass) in (1..).zip(masses) {
                lmp.command(&format!("mass {} {}", i, mass))?;
            }

            lmp.commands(pair_commands)?;
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
    pub fn set_structure(&mut self, new: Structure<P::Meta>) -> Result<()>
    {Ok({
        *self.structure.get_mut() = new;
    })}

    pub fn set_carts(&mut self, new: &[[f64; 3]]) -> Result<()>
    {Ok({
        self.structure.get_mut().set_carts(new.to_vec());
    })}

    pub fn set_lattice(&mut self, new: Lattice) -> Result<()>
    {Ok({
        self.structure.get_mut().set_lattice(&new);
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
    fn update_computation(&mut self) -> Result<()>
    {Ok({
        if self.structure.is_dirty() {
            self.structure.get_mut().ensure_carts();

            self.check_data_set_in_stone(self.structure.get())?;

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
            if self.structure.is_function_dirty(|s| self.potential.atom_types(s)) {
                self.send_lmp_types()?;
            }

            if self.structure.is_projection_dirty(|s| s.lattice()) {
                self.send_lmp_lattice()?;
            }

            if self.structure.is_projection_dirty(|s| s.as_carts_cached().unwrap()) {
                self.send_lmp_carts()?;
            }

            self.ptr.borrow_mut().command("run 0")?;
            self.structure.mark_clean();
        }
    })}

    fn send_lmp_types(&mut self) -> Result<()>
    {Ok({
        let meta = self.potential.atom_types(self.structure.get());
        assert_eq!(meta.len(), self.ptr.borrow_mut().get_natoms());

        let meta = meta.into_iter().map(AtomType::value).collect::<Vec<_>>();

        unsafe { self.ptr.borrow_mut().scatter_atoms_i("type", &meta) }?;
    })}

    fn send_lmp_carts(&mut self) -> Result<()>
    {Ok({
        let carts = self.structure.get().to_carts();
        assert_eq!(carts.len(), self.ptr.borrow_mut().get_natoms());

        unsafe { self.ptr.borrow_mut().scatter_atoms_f("x", carts.flat()) }?;
    })}

    fn send_lmp_lattice(&mut self) -> Result<()>
    {Ok({

        // From the documentation on 'change_box command':
        //
        //     Because the keywords used in this command are applied one at a time
        //     to the simulation box and the atoms in it, care must be taken with
        //     triclinic cells to avoid exceeding the limits on skew after each
        //     transformation in the sequence. If skew is exceeded before the final
        //     transformation this can be avoided by changing the order of the sequence,
        //     or breaking the transformation into two or more smaller transformations.
        //
        // There is nothing I can really say here that would not come across
        // as terribly, terribly rude. - ML

        let cur = self.structure.get().lattice().matrix();
        assert_eq!(0f64, cur[0][1], "non-triangular lattices not yet supported");
        assert_eq!(0f64, cur[0][2], "non-triangular lattices not yet supported");
        assert_eq!(0f64, cur[1][2], "non-triangular lattices not yet supported");

        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        enum Elem { Diagonal(&'static str), OffDiag(&'static str) }
        impl Elem {
            // tests whether a change to a given lattice matrix element
            //  could possibly cause us to crash if performed too soon.
            // This simple scheme allows us to handle MOST cases.
            pub fn should_defer(&self, old: f64, new: f64) -> bool
            { match *self {
                Elem::Diagonal(_) => new < old,
                Elem::OffDiag(_) => old.abs() < new.abs(),
            }}

            pub fn format(&self, value: f64) -> String
            { match *self {
                Elem::Diagonal(name) => format!("{} final 0 {}", name, value),
                Elem::OffDiag(name) => format!("{} final {}", name, value),
            }}
        }

        // NOTE: The order that these are declared are the order they will
        //       be set when no previous lattice has been sent.
        //       This order is safe for the garbage lattice with zero skew.
        let elems = &[
            (Elem::Diagonal("x"), (0, 0)),
            (Elem::Diagonal("y"), (1, 1)),
            (Elem::Diagonal("z"), (2, 2)),
            (Elem::OffDiag("xy"), (1, 0)),
            (Elem::OffDiag("xz"), (2, 0)),
            (Elem::OffDiag("yz"), (2, 1)),
        ];

        // Tragically, because we need to recall the last matrix that was set
        //  in order to determine what operations would cause lammps to crash,
        // we do need to match on whether or not a lattice has been saved,
        // which is exactly the sort of thing that MaybeDirty was supposed to
        // help prevent against.
        match self.structure.last_clean() {
            None => {
                let commands: Vec<_> =
                    elems.iter().map(|&(elem, (r, c))| {
                        format!("change_box all {}", elem.format(cur[r][c]))
                    }).collect();
                self.ptr.borrow_mut().commands(&commands)?;
            },

            Some(last) => {
                let last = last.lattice().matrix();
                // Look for cases that would defeat our simple ordering scheme.
                // (these are cases where both a diagonal and an off-diagonal would
                //   be classified as "defer until later", which is not enough information)
                // An example would be simultaneously skewing and shrinking a box.
                for r in 0..3 {
                    for c in 0..3 {
                        if Elem::Diagonal("").should_defer(cur[c][c], last[c][c]) &&
                            Elem::OffDiag("").should_defer(cur[r][c], last[r][c])
                        {
                            bail!("Tragically, you cannot simultaneously decrease a lattice \
                                diagonal element while increasing one of its off-diagonals. \
                                I'm sorry. You just can't.");
                        }
                    }
                }

                let mut do_early = vec![];
                let mut do_later = vec![];
                elems.iter().for_each(|&(elem, (r, c))| {
                    match elem.should_defer(last[r][c], cur[r][c]) {
                        true => { do_later.push((elem, (r, c))); },
                        false => { do_early.push((elem, (r, c))); },
                    }
                });

                let commands: Vec<_> =
                    None.into_iter()
                    .chain(do_early)
                    .chain(do_later)
                    .map(|(elem, (r, c))| {
                        format!("change_box all {}", elem.format(cur[r][c]))
                    })
                    .collect();

                self.ptr.borrow_mut().commands(&commands)?
            },
        }
    })}

    // Some of these properties probably could be allowed to change,
    // but it's not important enough for me to look into them right now,
    // so we simply check that they don't change.
    fn check_data_set_in_stone(&self, structure: &Structure<P::Meta>) -> Result<()>
    {Ok({
        fn check<D>(msg: &'static str, was: D, now: D) -> Result<()>
        where D: fmt::Debug + PartialEq,
        {
            ensure!(
                was == now,
                format!("{} changed since build()\n was: {:?}\n now: {:?}", msg, was, now));
            Ok(())
        }
        let InitInfo { masses, pair_commands } = self.potential.init_info(structure);
        let Lammps {
            original_num_atoms,
            original_init_info: InitInfo {
                masses: ref original_masses,
                pair_commands: ref original_pair_commands,
            },
            ..
        } = *self;

        check("number of atom types has", original_masses.len(), masses.len())?;
        check("number of atoms has", original_num_atoms, structure.num_atoms())?;
        check("masses have", original_masses, &masses)?;
        check("pair potential commands have", original_pair_commands, &pair_commands)?;
    })}
}

//-------------------------------------------
// gathering output from lammps
//
// NOTE: Every method here should call update_computation().
//       Don't worry about being sloppy with redundant calls;
//       the method was designed for such usage.
impl<P: Potential> Lammps<P> {

    pub fn compute(&mut self) -> Result<(f64, Vec<[f64; 3]>)>
    {Ok({
        self.update_computation()?;

        (self.compute_value()?, self.compute_grad()?)
    })}

    pub fn compute_value(&mut self) -> Result<f64>
    {Ok({
        self.update_computation()?;

        unsafe { self.ptr.borrow_mut().extract_compute_0d("RSP2_PE") }?
    })}

    pub fn compute_force(&mut self) -> Result<Vec<[f64; 3]>>
    {Ok({
        self.update_computation()?;

        let grad = unsafe { self.ptr.borrow_mut().gather_atoms_f("f", 3)? };
        grad.nest().to_vec()
    })}

    pub fn compute_grad(&mut self) -> Result<Vec<[f64; 3]>>
    {Ok({
        self.update_computation()?;

        let mut grad = self.compute_force()?;
        for x in grad.flat_mut() {
            *x *= -1.0;
        }
        grad
    })}

    pub fn compute_pressure(&mut self) -> Result<[f64; 6]>
    {Ok({
        self.update_computation()?;

        // as_array().clone() doesn't manage type inference here as well as deref...
        *unsafe {
            self.ptr.borrow_mut().extract_compute_1d("RSP2_Pressure", ComputeStyle::Global, 6)
        }?.as_array()
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

        fn atom_types(&self, structure: &Structure<()>) -> Vec<AtomType>
        { vec![AtomType::new(1); structure.num_atoms()]}

        fn init_info(&self, _: &Structure<()>) -> InitInfo
        { InitInfo {
            masses: vec![1.0],
            pair_commands: vec![PairCommand::pair_style("none")],
        }}
    }
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
        use ::rsp2_structure::Coords;
        let structure = Structure::new(
            Lattice::eye(),
            Coords::Fracs(vec![[0.0; 3]]),
            vec![()],
        );
        Builder::new().build(Default::default(), structure).unwrap()
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
        assert_matches!(
            Err(Error(ErrorKind::Lammps(Severity::Recoverable, _), _)),
            unsafe { lmp.ptr.borrow_mut().commands(&[
                // try to change to block with a nonzero skew
                "change_box all xy final 0.25",
                "change_box all ortho",
            ])}
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
