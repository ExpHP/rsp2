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
#[cfg(feature = "mpi")]
extern crate mpi as mpi_rs;

#[cfg(feature = "mpi")]
mod mpi {
    // There is little conceivable reason why rsmpi exposes a heirarchial public API
    // other than laziness; I see nothing in it that benefits from the namespacing,
    // and it adds a lot of mental overhead to remember the paths.
    pub(crate) use ::mpi_rs::{
        topology::*,
        collective::*,
        datatype::*,
        environment::*,
        point_to_point::*,
        raw::*,
    };
}

use ::maybe_dirty::MaybeDirty;
use ::low_level::{LowLevelApi, ComputeStyle, Skews, LammpsOwner};
#[cfg(feature = "mpi")]
use ::low_level::mpi::{MpiLammpsOwner, LammpsOnDemand as LammpsOnDemandImpl, LammpsDispatch};

use ::std::path::{Path, PathBuf};
use ::std::sync::{Mutex, MutexGuard};
use ::std::fmt;
use ::std::fs;
use ::std::io::Write;

use ::log::Level;
use ::slice_of_array::prelude::*;
use ::rsp2_structure::{Coords, Lattice};
use ::rsp2_array_types::{V3, Unvee, Envee};

pub type FailResult<T> = Result<T, ::failure::Error>;

pub use pub_types::*;
mod pub_types;

pub const API_TRACE_TARGET: &'static str = concat!(module_path!(), "::c_api");
pub const API_TRACE_LEVEL: Level = Level::Trace;

pub use ::low_level::{LammpsError, Severity};
mod low_level;

mod maybe_dirty;

lazy_static! {
    /// Guarantees that only one instance of Lammps may exist on a process,
    /// if constructed through safe APIs.
    pub static ref INSTANCE_LOCK: Mutex<InstanceLock> = Mutex::new(InstanceLock(()));
}

/// Proof that no instance of Lammps currently exists within the current process.
pub struct InstanceLock(());

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
    ptr: ::std::cell::RefCell<Box<dyn LowLevelApi>>,

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

    debug_dir: Option<PathBuf>,

    _lock: InstanceLockGuard,
}

//------------------------------------------

#[derive(Debug, Clone)]
pub struct Builder {
    make_instance: BoxDynMakeInstance,
    append_log: Option<PathBuf>,
    openmp_threads: Option<u32>,
    processors: [Option<u32>; 3],
    auto_adjust_lattice: bool,
    update_style: UpdateStyle,
    data_trace_dir: Option<PathBuf>,
    debug_dir: Option<PathBuf>,
    stdout: bool
}

trait MakeInstance: fmt::Debug {
    unsafe fn make_it(&self, argv: &[&str]) -> FailResult<Box<dyn LowLevelApi>>;
    fn box_clone(&self) -> BoxDynMakeInstance;
    // FIXME awkward to see this here
    fn eco_mode(&self, cont: &mut dyn FnMut());
}

// Cloneable wrapper
#[derive(Debug)]
pub struct BoxDynMakeInstance(Box<dyn MakeInstance + Send + Sync>);
impl Clone for BoxDynMakeInstance {
    fn clone(&self) -> Self { self.0.box_clone() }
}

#[derive(Debug, Clone)]
struct MakePlainInstance;

#[cfg(feature = "mpi")]
#[derive(Debug, Clone)]
struct MakeMpiInstance(LammpsOnDemandImpl);

impl MakeInstance for MakePlainInstance {
    unsafe fn make_it(&self, argv: &[&str]) -> FailResult<Box<dyn LowLevelApi>>
    { Ok(Box::new(LammpsOwner::new(argv)?)) }

    fn box_clone(&self) -> BoxDynMakeInstance
    { BoxDynMakeInstance(Box::new(self.clone())) }

    fn eco_mode(&self, cont: &mut dyn FnMut())
    { cont() }
}

#[cfg(feature = "mpi")]
impl MakeInstance for MakeMpiInstance {
    unsafe fn make_it(&self, argv: &[&str]) -> FailResult<Box<dyn LowLevelApi>>
    { Ok(Box::new(MpiLammpsOwner::new(self.0.clone(), argv)?)) }

    fn box_clone(&self) -> BoxDynMakeInstance
    { BoxDynMakeInstance(Box::new(self.clone())) }

    fn eco_mode(&self, cont: &mut dyn FnMut())
    { self.0.eco_mode(cont) }
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
// It only exists as a holdover from an earlier, less trivial design,
// and may eventually be removed.
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
        make_instance: MakePlainInstance.box_clone(),
        append_log: None,
        openmp_threads: None,
        update_style: UpdateStyle::safe(),
        auto_adjust_lattice: true,
        data_trace_dir: None,
        debug_dir: None,
        processors: [None; 3],
        stdout: false,
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

    /// Let lammps write directly to the standard output stream.
    ///
    /// The default value of `false` corresponds to the CLI arguments `-screen none`.
    pub fn stdout(&mut self, value: bool) -> &mut Self
    { self.stdout = value; self }

    /// Set the number of OpenMP threads per process.
    ///
    /// When `None`, the lammps default is used. (which is to use `OMP_NUM_THREADS`)
    pub fn openmp_threads(&mut self, value: Option<u32>) -> &mut Self
    { self.openmp_threads = value; self }

    /// Set the processor grid.
    ///
    /// This corresponds directly to LAMMPS' own `processors` command, although currently
    /// only simple grids are supported.  None stands for `*` in the original command syntax.
    pub fn processors(&mut self, value: [Option<u32>; 3]) -> &mut Self
    { self.processors = value; self }

    pub fn append_log(&mut self, path: impl AsRef<Path>) -> &mut Self
    { self.append_log = Some(path.as_ref().to_owned()); self }

    /// Enable MPI in Lammps from single-process mode.
    ///
    /// Afterwards, any Lammps potentials built through the builder will automatically
    /// coordinate the actions of the other processes.
    pub fn on_demand(&mut self, value: LammpsOnDemand) -> &mut Self
    { self._on_demand(value) }

    #[cfg(feature = "mpi")]
    fn _on_demand(&mut self, value: LammpsOnDemand) -> &mut Self
    { self.make_instance = MakeMpiInstance(value.imp).box_clone(); self }

    #[cfg(not(feature = "mpi"))]
    fn _on_demand(&mut self, value: LammpsOnDemand) -> &mut Self
    { match value {} }

    pub fn update_style(&mut self, value: UpdateStyle) -> &mut Self
    { self.update_style = value; self }

    pub fn eco_mode(&self, cont: &mut dyn FnMut())
    { self.make_instance.0.eco_mode(cont) }

    /// If set, an immensely large number of files will be written to this directory
    /// tracing the state of coordinate data both in rsp2 and LAMMPS.
    pub fn data_trace_dir(&mut self, value: Option<impl AsRef<Path>>) -> &mut Self
    { self.data_trace_dir = value.map(|p| p.as_ref().to_owned()); self }

    /// If set, some files may be written to this directory on error.
    ///
    /// The names use a fixed prefix.
    pub fn debug_dir(&mut self, value: Option<impl AsRef<Path>>) -> &mut Self
    { self.debug_dir = value.map(|p| p.as_ref().to_owned()); self }

    /// Call out to the LAMMPS C API to create an instance of Lammps,
    /// and configure it according to this builder.
    ///
    /// # Modifying the structure post-build
    ///
    /// `Lammps` has a number of methods that allow you to modify the structure
    /// after it has been built.  This can be significantly faster for running
    /// multiple computations than it would be to build an instance per structure.
    ///
    /// Ideally, modifying the structure after `build` should result in the same
    /// physical quantities as if you were to supply the modified structure from the start.
    /// But this is difficult to guarantee, and small numerical disturbances may arise.
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
    pub fn build<P>(
        &self,
        lock: InstanceLockGuard,
        potential: P,
        initial_coords: Coords,
        initial_meta: P::Meta,
    ) -> FailResult<Lammps<P>>
    where P: Potential,
    { Lammps::from_builder(self, lock, potential, initial_coords, initial_meta) }

    /// Create a `DispFn`, an alternative to `Lammps` which is optimized for computing
    /// forces at displacements.
    pub fn build_disp_fn<P>(
        &self,
        lock: InstanceLockGuard,
        potential: P,
        equilibrium_coords: Coords,
        equilibrium_meta: P::Meta,
    ) -> FailResult<DispFn<P>>
    where P: Potential,
    { DispFn::from_builder(self, lock, potential, equilibrium_coords, equilibrium_meta) }
}

/// A Lammps is built directly with the MutexGuard wrapper (rather than a reference)
/// to dodge an extra lifetime parameter.
pub type InstanceLockGuard = MutexGuard<'static, InstanceLock>;

/// Initialize LAMMPS, do nothing of particular value, and exit.
///
/// For debugging linker errors.
pub fn link_test() -> FailResult<()>
{Ok({
    let _ = unsafe { LammpsOwner::new(&["lammps", "-log", "none"])? };
})}

/// Initialize LAMMPS, do nothing of particular value, and exit.
///
/// For debugging linker errors.
#[cfg(feature = "mpi")]
pub fn mpi_link_test() -> FailResult<()>
{Ok({
    println!("{}", ::mpi::library_version().unwrap());
    LammpsOnDemandImpl::install(
        LammpsDispatch::new(),
        |on_demand| {
            unsafe { MpiLammpsOwner::new(
                on_demand,
                &["lammps", "-log", "none"],
            )}.map(drop)
        },
    );
})}

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

// FIXME: Doc.
//
// This is just an opaque type that serves as the publically visible form of MpiOnDemand.
// Most of the docs for MpiOnDemand apply directly to this, but I'm tired of repeating myself...
#[cfg(feature = "mpi")]
pub struct LammpsOnDemand { imp: LammpsOnDemandImpl }

#[cfg(not(feature = "mpi"))]
pub enum LammpsOnDemand {}

#[cfg(feature = "mpi")]
impl LammpsOnDemand {
    /// Continuation-style constructor.
    ///
    /// # MPI
    ///
    /// This is to be called on all MPI processes.
    ///
    /// The continuation runs on a single process, while the others run in an event loop
    /// until the continuation finishes. MPI will be unusable for anything other LAMMPS during
    /// this time.
    ///
    /// **Important:** Currently, if a process panics at a bad time, the other processes may
    /// get deadlocked on a blocking operation. Nothing is done about this by default.
    /// See the helper method `with_mpi_abort_on_unwind` for a possible solution.
    ///
    /// # Output
    ///
    /// `Some` on the root process, `None` on the others.
    ///
    /// # Panics
    ///
    /// Panics if the LammpsOnDemand is leaked outside the continuation.
    pub fn install<R>(continuation: impl FnOnce(LammpsOnDemand) -> R) -> Option<R> {
        LammpsOnDemandImpl::install(
            LammpsDispatch::new(),
            |on_demand| continuation(LammpsOnDemand { imp: on_demand }),
        )
    }

    /// Helper to call `MPI_ABORT` if a panic occurs inside the continuation,
    /// *after* allowing the panic implementation to unwind back out.
    ///
    /// This will be completely ineffective if the panic implementation does not unwind.
    pub fn with_mpi_abort_on_unwind<R>(func: impl ::std::panic::UnwindSafe + FnOnce() -> R) -> R {
        ::low_level::mpi_helper::with_mpi_abort_on_unwind(func)
    }
}

//-------------------------------------------
// Initializing the LAMMPS C API object.
//
impl<P: Potential> Lammps<P>
{
    // implementation of Builder::Build
    fn from_builder(builder: &Builder, lock: InstanceLockGuard, potential: P, coords: Coords, meta: P::Meta) -> FailResult<Self>
    {Ok({
        let original_num_atoms = coords.num_atoms();
        let original_init_info = potential.init_info(&coords, &meta);
        let original_atom_types = potential.atom_types(&coords, &meta);

        let ptr = Self::_from_builder(builder, &original_init_info, &coords, &original_atom_types)?;
        Lammps {
            ptr: ::std::cell::RefCell::new(ptr),
            structure: MaybeDirty::new_dirty((coords, meta)),
            potential,
            original_init_info,
            original_num_atoms,
            auto_adjust_lattice: builder.auto_adjust_lattice,
            update_fsm: builder.update_style.initial_fsm(),
            data_trace_dir: builder.data_trace_dir.clone(),
            debug_dir: builder.debug_dir.clone(),
            _lock: lock,
        }
    })}

    // monomorphic
    fn _from_builder(
        builder: &Builder,
        init_info: &InitInfo,
        coords: &Coords,
        atom_types: &[AtomType],
    ) -> FailResult<Box<dyn LowLevelApi>>
    {Ok({
        use ::std::io::prelude::*;

        let mut lmp = {
            let mut argv = vec![
                "lammps",
                "-log", "none", // logs opened from CLI are truncated, but we want to append
            ];
            if !builder.stdout {
                argv.push("-screen");
                argv.push("none");
            }

            // NOTE: safe due to how the only callers use the instance lock
            unsafe { builder.make_instance.0.make_it(&argv)? }
        };

        // Append a header to the log file as a feeble attempt to help delimit individual
        // runs (even though it will still get messy for parallel runs).
        //
        // NOTE: This looks like a surprising hidden side-effect, but it isn't really.
        //       Or rather, that is to say, removing it won't make things any better,
        //       because LAMMPS itself will be writing many things to this same file
        //       anyways over the course of this function.
        append_logfile_nonessential(builder.append_log.as_ref(), |mut f| {
            let _ = writeln!(f, "---------------------------------------------");
            let _ = writeln!(f, "---- Begin run at {}", ::chrono::Local::now());
            let _ = writeln!(f, "---------------------------------------------");
        });

        if let Some(log_file) = &builder.append_log {
            lmp.command(
                format!("log {} append", log_file.display()),
            )?;
        }

        lmp.command(format!("package omp {}", builder.openmp_threads.unwrap_or(0)))?;
        lmp.command(format!("processors {}", {
            builder.processors.iter()
                .map(|x| match x {
                    None => "*".to_string(),
                    Some(x) => x.to_string(),
                })
                .collect::<Vec<_>>().join(" ")
        }))?;

        lmp.commands(&[
            "units metal",                  // Angstroms, picoseconds, eV
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
            "region sim prism 0 2 0 2 0 2 0 0 0", // garbage garbage garbage
        ])?;

        {
            let InitInfo { masses, pair_style, pair_coeffs } = init_info;

            lmp.command(format!("create_box {} sim", masses.len()))?;
            for (i, mass) in (1..).zip(masses) {
                lmp.command(format!("mass {} {}", i, mass))?;
            }

            lmp.command(pair_style.to_string())?;
            lmp.commands(pair_coeffs)?;
        }

        // HACK:
        //   set positions explicitly using commands to resolve "lost atoms" errors
        //   when using certain MPI node layouts on certain structures.
        //
        //   I was trying to avoid this, but it seems that certain information about which atoms
        //   belong to which nodes is only updated when certain commands are invoked; the initial
        //   binning update from 'pre yes' on the first iteration does not suffice.
        {
            send_lmp_lattice(&mut *lmp, coords.lattice(), builder.auto_adjust_lattice)?;

            // Setting individual positions spams the logfile.
            // Until we find a better way, just stop writing to it momentarily.
            lmp.command("log none".to_string())?;
            append_logfile_nonessential(builder.append_log.as_ref(), |mut f| {
                let _ = writeln!(f, "Setting positions explicitly using the 'create_atom single' command.");
                let _ = writeln!(f, "(commands omitted from logfile)");
            });

            lmp.init_atoms(
                coords.to_carts().unvee(),
                atom_types.iter().map(|ty| ty.value()).collect(),
            )?;

            // Resume logging.
            if let Some(log_file) = &builder.append_log {
                lmp.command(
                    format!("log {} append", log_file.display()),
                )?;
            }
        }

        // set up computes
        lmp.commands(&[
            &format!("compute RSP2_PE all pe"),
            &format!("compute RSP2_Pressure all pressure NULL virial"),
        ])?;

        lmp
    })}
}

/// Tries to temporarily open the logfile to append something non-essential
/// (like a divider or placeholder). IO errors are silently ignored.
fn append_logfile_nonessential(
    path: Option<impl AsRef<Path>>,
    cont: impl FnOnce(::std::fs::File),
) {
    if let Some(path) = path {
        let log = {
            ::std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .append(true)
                .open(path)
        };
        if let Ok(file) = log {
            cont(file)
        }
    }
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

        self.ptr.borrow_mut().command(command.into())?;

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

        let types = types.into_iter().map(AtomType::value).collect();

        unsafe { self.ptr.borrow_mut().scatter_atoms_i("type".into(), types) }?;
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
        unsafe { self.ptr.borrow_mut().scatter_atoms_f("x".into(), new_lmp_carts.unvee_ref().flat().to_vec()) }?;
    })}

    fn send_lmp_lattice(&mut self) -> FailResult<()>
    { send_lmp_lattice(
        &mut **self.ptr.borrow_mut(),
        self.structure.get().0.lattice(),
        self.auto_adjust_lattice,
    )}

    // Some of these properties probably could be allowed to change,
    // but it's not important enough for me to look into them right now,
    // so we simply check that they don't change.
    fn check_data_set_in_stone(&self, coords: &Coords, meta: &P::Meta) -> FailResult<()>
    {Ok({
        // struct acting as a generic closure
        struct Closure<'a, P: Potential + 'a> {
            lmp: &'a Lammps<P>,
            coords: &'a Coords,
        }
        impl<'a, P: Potential + 'a> Closure<'a, P> {
            fn check<D>(&self, msg: &'static str, was: D, now: D) -> FailResult<()>
            where D: fmt::Debug + PartialEq,
            {
                let Closure { lmp, coords } = *self;
                if was != now {
                    if let Some(mut file) = lmp.create_debug_dir_file("lammps-wrap-debug-carts") {
                        let _ = writeln!(file, "{:?}", coords.to_carts());
                    }
                    bail!("{} changed since build()\n was: {:?}\n now: {:?}", msg, was, now);
                }
                Ok(())
            }
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

        let c = Closure { lmp: self, coords };
        c.check("number of atom types has", original_masses.len(), masses.len())?;
        c.check("number of atoms has", original_num_atoms, coords.num_atoms())?;
        c.check("masses have", original_masses, &masses)?;
        c.check("pair_style command has", original_pair_style, &pair_style)?;
        c.check("pair_coeff commands have", original_pair_coeffs, &pair_coeffs)?;
    })}

    fn write_data_trace_fileset(&self, dir: &Path, filename_prefix: &str) {
        let _ = self._write_data_trace_fileset(dir, filename_prefix);
    }

    fn _write_data_trace_fileset(&self, dir: &Path, filename_prefix: &str) -> FailResult<()>
    {Ok({
        fs::create_dir_all(dir)?;
        let file = |ext: &str| {
            fs::File::create(dir.join(format!("{}.{}", filename_prefix, ext)))
        };
        writeln!(file("lmp.carts")?, "{:?}", self.read_raw_lmp_carts()?)?;
        writeln!(file("lmp.force")?, "{:?}", self.read_raw_lmp_force()?)?;
        writeln!(file("cached.carts")?, "{:?}", self.structure.get().0.to_carts())?;
    })}

    fn create_debug_dir_file(&self, name: &str) -> Option<fs::File> {
        if let Some(ref debug_dir) = self.debug_dir {
            let path = debug_dir.join(name);
            match fs::File::create(&path) {
                Ok(file) => {
                    info!("wrote {}", path.display());
                    Some(file)
                },
                Err(e) => {
                    warn!("could not create lammps-wrap debug file {}: {}", path.display(), e);
                    None
                },
            }
        } else {
            None
        }
    }
}

fn send_lmp_lattice(lmp: &mut (dyn LowLevelApi + 'static), lattice: &Lattice, auto_adjust: bool) -> FailResult<()>
{Ok({
    let [
    [xx, _0, _1],
    [xy, yy, _2],
    [xz, yz, zz],
    ] = lattice.matrix().unvee();

    assert_eq!(0f64, _0, "non-triangular lattices not yet supported");
    assert_eq!(0f64, _1, "non-triangular lattices not yet supported");
    assert_eq!(0f64, _2, "non-triangular lattices not yet supported");

    let mut diag = [xx, yy, zz];
    let mut skews = Skews { xy, yz, xz };
    if auto_adjust {
        auto_adjust_lattice(&mut diag, &mut skews);
    }

    // safe because we initialized the lattice the hard way during initialization,
    // so surely `domain->set_initial_box()` must have been called...
    unsafe { lmp.reset_box([0.0, 0.0, 0.0], diag, skews)? }

    // "Is this box to your liking, sire?"
    lmp.commands(&[
        // These should have no effect, but they give LAMMPS a chance to look at the
        // box and throw an exception if it doesn't like what it sees.
        format!("change_box all x final 0 {}", diag[0]),
        format!("change_box all xy final {}", skews.xy),
    ])?;
})}

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
        let x = unsafe { self.ptr.borrow_mut().gather_atoms_f("x".into(), 3)? };
        x.nest::<[_; 3]>().to_vec().envee()
    })}

    fn read_raw_lmp_force(&self) -> FailResult<Vec<V3>>
    {Ok({
        let x = unsafe { self.ptr.borrow_mut().gather_atoms_f("f".into(), 3)? };
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

        unsafe { self.ptr.borrow_mut().extract_compute_0d("RSP2_PE".into()) }?
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
            self.ptr.borrow_mut().extract_compute_1d("RSP2_Pressure".into(), ComputeStyle::Global, 6)
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
    fn from_builder(builder: &Builder, lock: InstanceLockGuard, potential: P, coords: Coords, meta: P::Meta) -> FailResult<Self>
    {Ok({
        let mut builder = builder.clone();
        builder.update_style(UpdateStyle { n: 1, pre: false, post: false, sync_positions_every: 1 });

        let mut lammps = Lammps::from_builder(&builder, lock, potential, coords, meta)?;

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
        Builder::new().build(INSTANCE_LOCK.lock().unwrap(), Default::default(), coords, ()).unwrap()
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
