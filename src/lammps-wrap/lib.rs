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

macro_rules! zip_eq {
    ($a:expr, $b:expr) => {{
        let a = $a.into_iter();
        let b = $b.into_iter();
        assert_eq!(a.len(), b.len());
        a.zip(b)
    }}
}

pub type StdResult<T, E> = ::std::result::Result<T, E>;

use ::low_level::ComputeStyle;
pub use ::low_level::Severity;
use low_level::LammpsOwner;
mod low_level;

use ::std::path::{Path, PathBuf};
use ::slice_of_array::prelude::*;
use ::rsp2_structure::{Structure, Lattice};

const REBO_MASS_CARBON:   f64 = 12.01;
const REBO_MASS_HYDROGEN: f64 =  1.00;

pub struct Lammps<P> {
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
    structure: MaybeDirty<Structure<AtomType>>,
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
}

#[derive(Debug, Clone)]
pub struct Builder {
    append_log: Option<PathBuf>,
    lj_strength: f64,
    lj_sigma: f64,
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
        lj_strength: 1.0,
        lj_sigma: 3.0,
    }}

    pub fn append_log<P: AsRef<Path>>(&mut self, path: P) -> &mut Self
    { self.append_log = Some(path.as_ref().to_owned()); self }

    pub fn threaded(&mut self, value: bool) -> &mut Self
    { self.threaded = value; self }

    pub fn lj_strength(&mut self, value: f64) -> &mut Self
    { self.lj_strength = value; self }

    pub fn lj_sigma(&mut self, value: f64) -> &mut Self
    { self.lj_sigma = value; self }

    pub fn build<P>(&self, potential: P, structure: Structure<P::Meta>) -> Result<Lammps<P>>
    where P: Potential,
    { Lammps::from_builder(self, potential, structure) }

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

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct AtomType(i64);
impl AtomType {
    pub fn new(x: i64) -> Self {
        assert!(x > 0);
        AtomType(x as _)
    }
    pub fn int(self) -> i64 { self.0 }

    pub fn slice_as_ints(slice: &[Self]) -> &[i64]
    {
        assert_eq!(::std::mem::size_of::<Self>(), ::std::mem::size_of::<i64>());
        assert_eq!(::std::mem::align_of::<Self>(), ::std::mem::align_of::<i64>());
        unsafe { ::std::slice::from_raw_parts(slice.as_ptr() as *const _, slice.len()) }
    }
}

// Code that abstracts over the different commands we need
// to send based on which potential we are using.
pub use potential::{Potential, Airebo, KolmogorovCrespi};

use potential::InitInfo;
mod potential {
    use super::*;
    use ::std::fmt::Debug;
    use rsp2_structure::Element;
    use rsp2_structure::Layer;

    // ----------------------------
    // Abstract interface

    /// Represents a pair potential of Lammps that is explicitly supported by RSP2.
    ///
    /// The Lammps type is parameterized over one of these.
    pub trait Potential {
        /// User-facing atom metadata type.
        ///
        /// The public API of `Lammps<Self>` takes structures of type `Structure<Self::Meta>`
        type Meta: Debug;

        // Conversions to and fro.
        // Conversions from meta may produce user-facing errors.
        #[doc(hidden)] fn _atom_type_from_meta(&self, meta: &Self::Meta) -> Result<AtomType>;
        #[doc(hidden)] fn _meta_from_atom_type(&self, meta: AtomType) -> Self::Meta;

        // See InitInfo for an explanation.
        #[doc(hidden)] fn _lammps_setup_info(&self, structure: &Structure<Self::Meta>) -> Result<InitInfo>;

        // Produces the type of structure stored in the `Lammps` object.
        #[doc(hidden)] fn _convert_structure(&self, structure: Structure<Self::Meta>) -> Result<Structure<AtomType>>
        {Ok({
            let meta = structure.metadata().iter()
                    .map(|m| self._atom_type_from_meta(m))
                    .collect::<Result<Vec<_>>>()?;
            structure.with_metadata(meta)
        })}
    }

    /// Data describing the commands which need to be sent to lammps to initialize
    /// atom types and the potential, for a specific Potential and Structure.
    #[doc(hidden)]
    pub struct InitInfo {
        /// Mass of each atom type.
        pub(crate) masses: Vec<f64>,
        /// Contains one or more of the following commands:
        ///
        /// * pair_style
        /// * pair_coeff
        pub(crate) pair_commands: Vec<String>,
    }

    impl InitInfo {
        pub(crate) fn atom_type_ids(&self) -> Box<ExactSizeIterator<Item=AtomType>>
        { Box::new((0..self.masses.len()).map(|i| AtomType::new(i as i64 + 1))) }
    }

    // ----------------------------
    // Implementations

    /// Uses `pair_style airebo`.
    #[derive(Debug, Clone)]
    pub struct Airebo {
        pub lj_sigma: f64,
        pub lj_strength: f64,
        pub lj: bool,
        pub torsion: bool,
    }

    impl Default for Airebo {
        fn default() -> Self
        { Airebo {
            lj_sigma: 3.0,
            lj_strength: 1.0,
            lj: true,
            torsion: false,
        }}
    }

    impl Airebo {
        pub fn new() -> Self { Self::default() }
        pub fn lj_sigma(&mut self, value: f64) -> &mut Self { self.lj_sigma = value; self }
        pub fn lj_strength(&mut self, value: f64) -> &mut Self { self.lj_strength = value; self }
        pub fn lj_enabled(&mut self, value: bool) -> &mut Self { self.lj = value; self }
        pub fn torsion_enabled(&mut self, value: bool) -> &mut Self { self.torsion = value; self }
    }

    impl Potential for Airebo {
        type Meta = Element;

        fn _atom_type_from_meta(&self, meta: &Self::Meta) -> Result<AtomType>
        { match meta.symbol() {
            "H" => Ok(AtomType::new(1)),
            "C" => Ok(AtomType::new(2)),
            _ => bail!(ErrorKind::BadMeta("Airebo", format!("{:?}", meta))),
        }}

        fn _meta_from_atom_type(&self, id: AtomType) -> Self::Meta
        { match id.int() {
            1 => Element::from_symbol("H").unwrap(),
            2 => Element::from_symbol("C").unwrap(),
            _ => panic!("unexpected atom type from lammps: {}", id.int()),
        }}

        fn _lammps_setup_info(&self, _: &Structure<Self::Meta>) -> Result<InitInfo>
        { Ok(InitInfo {
            masses: vec![REBO_MASS_HYDROGEN, REBO_MASS_CARBON],
            pair_commands: vec![
                format!(
                    "pair_style airebo/omp {} {} {}",
                    self.lj_sigma, boole(self.lj), boole(self.torsion),
                ),
                format!("pair_coeff * * CH.airebo H C"), // read potential info
                format!("pair_coeff * * lj/scale {}", self.lj_strength), // set lj potential scaling factor (HACK)
            ],
        })}
    }

    fn boole(b: bool) -> u32 { b as _ }

    /// Uses `pair_style kolmogorov/crespi/z`.
    #[derive(Debug, Clone, Default)]
    pub struct KolmogorovCrespi {
        cutoff: f64,
    }

    impl KolmogorovCrespi {
        pub fn new() -> Self { Self::default() }
    }

    impl Potential for KolmogorovCrespi {
        type Meta = Layer;

        fn _atom_type_from_meta(&self, &Layer(layer): &Self::Meta) -> Result<AtomType>
        { Ok(AtomType::new(layer as i64 + 1)) }

        fn _meta_from_atom_type(&self, id: AtomType) -> Self::Meta
        { Layer(id.int() as u32 - 1) }

        fn _lammps_setup_info(&self, structure: &Structure<Self::Meta>) -> Result<InitInfo>
        {
            let nlayers = structure.metadata().iter().map(|&Layer(layer)| layer).max().unwrap_or(0) + 1;
            let masses = vec![REBO_MASS_CARBON; nlayers as usize];

            let mut pair_commands = vec![
                format!("pair_style hybrid/overlay rebo kolmogorov/crespi/z {}", self.cutoff),
                format!("pair_coeff * * rebo                 CH.airebo  C C"),
            ];
            pair_commands.extend((0..nlayers - 1).map(|i| {
                let typ = i + 1;
                format!("pair_coeff {} {} kolmogorov/crespi/z  CC.KC      C C", typ, typ + 1)
            }));

            Ok(InitInfo { masses, pair_commands })
        }
    }
}

impl<P: Potential> Lammps<P>
{
    fn from_builder(builder: &Builder, potential: P, structure: Structure<P::Meta>) -> Result<Self>
    {Ok({
        let info_from_potential = potential._lammps_setup_info(&structure)?;
        let structure = potential._convert_structure(structure)?;

        let ptr = Self::_from_builder(builder, structure.num_atoms(), info_from_potential)?;
        Lammps {
            ptr: ::std::cell::RefCell::new(ptr),
            structure: MaybeDirty::new_dirty(structure),
            potential,
        }
    })}

    // monomorphic
    fn _from_builder(
        builder: &Builder,
        num_atoms: usize,
        info_from_potential: InitInfo,
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
            let atom_types = info_from_potential.atom_type_ids();
            let n_atom_types = atom_types.len();

            lmp.command(&format!("create_box {} sim", n_atom_types))?;
            for (i, mass) in zip_eq!(atom_types, info_from_potential.masses) {
                lmp.command(&format!("mass {} {}", i.int(), mass))?;
            }

            lmp.commands(&info_from_potential.pair_commands[..])?;
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
// modifying the system
//
// All these need to do is modify the Structure object through the dirtying API.
//
// There's nothing tricky that needs to be done here to ensure that updates are
// propagated to LAMMPS, so long as `update_computation` is able to correctly
// detect when the new values differ from the old.
impl<P: Potential> Lammps<P> {
    pub fn set_structure(&mut self, structure: Structure<P::Meta>) -> Result<()>
    {Ok({
        let new = self.potential._convert_structure(structure)?;
        *self.structure.get_mut() = new;
    })}

    pub fn set_metadata(&mut self, meta: &[P::Meta]) -> Result<()>
    where P::Meta: Clone,
    {Ok({
        let new: Vec<_> = meta.iter().map(|m| self.potential._atom_type_from_meta(m)).collect::<Result<_>>()?;
        self.structure.get_mut().set_metadata(new);
    })}

    pub fn set_carts(&mut self, carts: &[[f64; 3]]) -> Result<()>
    {Ok({
        self.structure.get_mut().set_carts(carts.to_vec());
    })}

    pub fn set_lattice(&mut self, lattice: Lattice) -> Result<()>
    {Ok({
        self.structure.get_mut().set_lattice(&lattice);
    })}
}

//-------------------------------------------
// sending input to lammps and running the main computation

impl<P> Lammps<P> {
    // This will rerun computations in lammps, but only if things have changed.
    // (The very first time it is called is also when the everything in LAMMPS
    //  will be populated with initial data that isn't just garbage.)
    //
    // At the end, (cached, updated) == (Some(_), None)
    fn update_computation(&mut self) -> Result<()>
    {Ok({
        if self.structure.is_dirty() {
            self.structure.get_mut().ensure_carts();

            // Only send data that has changed from the cache.
            // This is done because it appears that lammps does some form of
            // caching as well (lattice modifications in particular appear
            // to increase the amount of computational work)
            //
            // The first time through, all projections will be considered dirty,
            // so everything will be sent to replace the garbage data that we
            // gave lammps during initialization.
            if self.structure.is_projection_dirty(|s| s.metadata()) {
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
        let meta = AtomType::slice_as_ints(self.structure.get().metadata());

        assert_eq!(meta.len(), self.ptr.borrow_mut().get_natoms());

        unsafe { self.ptr.borrow_mut().scatter_atoms_i("type", meta) }?;
    })}

    fn send_lmp_carts(&mut self) -> Result<()>
    {Ok({
        let carts = self.structure.get().to_carts();
        let natoms = self.ptr.borrow_mut().get_natoms();
        assert_eq!(natoms, carts.len());

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
}

//-------------------------------------------
// gathering output from lammps
//
// NOTE: Every method here should call update_computation().
//       Don't worry about being sloppy with redundant calls;
//       the method was designed for such usage.
impl<P> Lammps<P> {

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

//-------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: this is now mostly tested indirectly
    //       through the other crates that use it.

    // get a fresh Lammps instance on which arbitrary functions can be called.
    fn arbitrary_initialized_lammps() -> Lammps<Airebo>
    {
        use ::rsp2_structure::Coords;
        use ::rsp2_structure::consts::CARBON;
        let structure = Structure::new(
            Lattice::eye(),
            Coords::Fracs(vec![[0.0; 3]]),
            vec![CARBON],
        );
        Builder::new().build(Airebo::default(), structure).unwrap()
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
