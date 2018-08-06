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

//================================================================================
//
//                A guide to what is happening in this module
//                  (because unfortunately, you'll need it!)
//
// Basically, the majority of the madness contained within this module is all to implement
// LowLevelApi for MpiLammpsOwner. When you call a method on MpiLammpsOwner,
// here's what happens:
//
// * Let's say you called `gather_atoms_i`:
//
//       mpi_lammps.gather_atoms_i(name, count)    // called on root process
//
// * The method body in the `LowLevelApi` impl in this module (which only runs on the root process)
//   creates an `Input` that encodes this specific method call at runtime:
//
//       Input::GatherAtomsI(InputGatherAtomsI { name, count })
//
//   and hands it to `mpi_on_demand.invoke`.
//
// * `mpi_on_demand.invoke` uses the Broadcast impl of `Input` to communicate the value to the other
//   processes.  Then it calls `LammpsDispatch`.
//
// * `LammpsDispatch` (which runs on all processes) uses the variant of Input to dispatch to
//   the appropriate method on the *plain* LammpsOwner.
//
//       plain_lammps.gather_atoms_i(name, count)   // called on all processes
//
//   The output unfortunately must be temporarily encoded in yet another enum:
//
//       Output::GatherAtomsI(out)
//
// * `mpi_on_demand.invoke` forgets the values returned on non-root processes,
//   and on root it returns the `Output` back to our implementation here of `gather_atoms_i`.
//
// * The implementation validates that the Output is the correct variant and returns it.
//
// .......got all that?
//
// To make matters worse, some of the dispatching magic is done in obscure ways using macros,
// in part to avoid having to repeatedly type the field names.
//
// ...anyways, this is the price we pay for using MpiOnDemand.  It's precisely what I signed up
// for, and I think it's worth it. The fact of the matter is that I almost never need to touch
// these low-level parts of the API, so this small little pit of maintenance hell is not a big deal.
//
//================================================================================

use ::mpi;
use ::FailResult;
use ::std::sync::{Arc, Mutex};
use ::low_level::{LammpsOwner, Skews, ComputeStyle, LowLevelApi};
use ::low_level::mpi_helper::{
    MpiOnDemand, Broadcast, DispatchMultiProcess, this_process_is_root,
};

/// A drop-in replacement for `LammpsOwner` which uses `MpiOnDemand`.
/// Only exists on the root process.
///
/// All methods are little more than wrappers around `LammpsOnDemand` that notify the other
/// processes about the method being called, and which handle the translation between the fixed
/// input/output types per method and the catch-all enums used by the event loop body.
///
/// A custom `Drop` impl notifies the other processes about the drop.
/// This type is expressly NOT CLONE.
#[cfg(feature = "_mpi")]
#[derive(Debug)]
pub(crate) struct MpiLammpsOwner<Root: mpi::Root> {
    on_demand: LammpsOnDemand<Root>,
}

pub(crate) type LammpsOnDemand<Root> = MpiOnDemand<Root, LammpsDispatch>;

impl<Root: mpi::Root> MpiLammpsOwner<Root> {
    /// Construct an `MpiLammpsOwner`.
    ///
    /// # MPI
    ///
    /// The usage of `MpiOnDemand` ensures that this is only called on the root process.
    ///
    /// # Safety
    ///
    /// Like `LammpsOwner`, construction of `MpiLammpsOwner` is inherently unsafe because it is
    /// unsafe to use multiple instances simultaneously on separate threads.
    #[cfg(feature = "_mpi")]
    pub(crate) unsafe fn new(
        on_demand: LammpsOnDemand<Root>,
        argv: &[&str],
    ) -> FailResult<Self>
    {Ok({
        let argv: Vec<_> = argv.iter().map(|s| s.to_string()).collect();

        match on_demand.invoke(Input::New(InputNew { argv })) {
            Output::New(res) => {
                // FIXME: If this fails, the event loop is probably totally borked, but nothing
                //        can stop the user from using it again.
                let () = res?;
            },
            _ => panic!("wrong output variant!"),
        }

        MpiLammpsOwner { on_demand }
    })}
}

impl<Root: mpi::Root> Drop for MpiLammpsOwner<Root> {
    fn drop(&mut self) {
        match self.on_demand.invoke(Input::Drop(InputDrop { })) {
            Output::Drop(()) => {},
            _ => panic!("wrong output variant!")
        }
    }
}

//------------------------------------------------

c_enums!{
    // Used to communicate the Input variant in the Broadcast impl.
    [] enum Method {
        New = 0,
        Drop = 1,
        Command = 2,
        GetNatoms = 3,
        ResetBox = 4,
        GatherAtomsI = 5,
        GatherAtomsF = 6,
        ScatterAtomsI = 7,
        ScatterAtomsF = 8,
        ExtractCompute0d = 9,
        ExtractCompute1d = 10,
    }
}

pub(crate) enum Input {
    // individual structs per enum to enable macro codegen in Broadcast impls
    New(InputNew),
    Drop(InputDrop),
    Command(InputCommand),
    GetNatoms(InputGetNatoms),
    ResetBox(InputResetBox),
    GatherAtomsI(InputGatherAtomsI),
    GatherAtomsF(InputGatherAtomsF),
    ScatterAtomsI(InputScatterAtomsI),
    ScatterAtomsF(InputScatterAtomsF),
    ExtractCompute0d(InputExtractCompute0d),
    ExtractCompute1d(InputExtractCompute1d),
}

pub(crate) enum Output {
    New(OutputNew),
    Drop(OutputDrop),
    Command(OutputCommand),
    GetNatoms(OutputGetNatoms),
    ResetBox(OutputResetBox),
    GatherAtomsI(OutputGatherAtomsI),
    GatherAtomsF(OutputGatherAtomsF),
    ScatterAtomsI(OutputScatterAtomsI),
    ScatterAtomsF(OutputScatterAtomsF),
    ExtractCompute0d(OutputExtractCompute0d),
    ExtractCompute1d(OutputExtractCompute1d),
}

// Generates broadcast impls that broadcast each field,
// and methods that call a method on LammpsOwner
macro_rules! derive_low_level_api {
    ($(
        #[$(($unsafe_keyword:tt))* fn $method_name:ident / $MethodName:ident() -> $OutputName:ident]
        pub(crate) struct $InputName:ident { $($field:ident : $Field:ty),* }
    )*) => {
        $(
            derive_broadcast! {
                pub(crate) struct $InputName { $($field : $Field),* }
            }

            impl $InputName {
                $($unsafe_keyword)* fn invoke_method(self, _: &impl mpi::Root, lammps: &mut Option<LammpsOwner>) -> $OutputName {
                    if let Some(lammps) = lammps {
                        let $InputName { $($field),* } = self;
                        lammps.$method_name($($field),*)
                    } else {
                        panic!("BUG: {} called before new", stringify!($method_name));
                    }
                }
            }
        )*

        impl<Root: mpi::Root> LowLevelApi for MpiLammpsOwner<Root> {
            $(
                $($unsafe_keyword)* fn $method_name(&mut self, $($field : $Field),*) -> $OutputName {
                    match self.on_demand.invoke(Input::$MethodName($InputName { $($field),* })) {
                        Output::$MethodName(out) => out,
                        _ => panic!("wrong output variant!")
                    }
                }
            )*
        }
    };
}

macro_rules! derive_broadcast {
    ($(
        pub(crate) struct $InputName:ident { $($field:ident : $Field:ty),* }
    )*) => {$(
        // Also generate the struct itself!
        /// This type exists to facilitate codegen.
        pub(crate) struct $InputName { $($field : $Field),* }

        impl Broadcast for $InputName {
            fn broadcast(_root: &impl mpi::Root, value: Option<Self>) -> Self {
                match value {
                    Some($InputName { $($field),* }) => {
                        $(
                            let $field = Broadcast::broadcast(_root, Some($field));
                        )*
                        $InputName { $($field),* }
                    },
                    None => {
                        $(
                            let $field = Broadcast::broadcast(_root, None);
                        )*
                        $InputName { $($field),* }
                    },
                }
            }
        }
    )*};
}

derive_broadcast! {
    pub(crate) struct InputNew { argv: Vec<String> }

    pub(crate) struct InputDrop { }
}

derive_low_level_api! {
    #[fn command/Command() -> OutputCommand]
    pub(crate) struct InputCommand { cmd: String }

    #[fn get_natoms/GetNatoms() -> OutputGetNatoms]
    pub(crate) struct InputGetNatoms { }

    #[(unsafe) fn reset_box/ResetBox() -> OutputResetBox]
    pub(crate) struct InputResetBox { low: [f64; 3], high: [f64; 3], skews: Skews }

    #[(unsafe) fn gather_atoms_i/GatherAtomsI() -> OutputGatherAtomsI]
    pub(crate) struct InputGatherAtomsI { name: String, count: usize }

    #[(unsafe) fn gather_atoms_f/GatherAtomsF() -> OutputGatherAtomsF]
    pub(crate) struct InputGatherAtomsF { name: String, count: usize }

    #[(unsafe) fn scatter_atoms_i/ScatterAtomsI() -> OutputScatterAtomsI]
    pub(crate) struct InputScatterAtomsI { name: String, data: Vec<i64> }

    #[(unsafe) fn scatter_atoms_f/ScatterAtomsF() -> OutputScatterAtomsF]
    pub(crate) struct InputScatterAtomsF { name: String, data: Vec<f64> }

    #[(unsafe) fn extract_compute_0d/ExtractCompute0d() -> OutputExtractCompute0d]
    pub(crate) struct InputExtractCompute0d { name: String }

    #[(unsafe) fn extract_compute_1d/ExtractCompute1d() -> OutputExtractCompute1d]
    pub(crate) struct InputExtractCompute1d { name: String, style: ComputeStyle, len: usize }
}

// New and Drop are special.
//
// They are included in the dispatch list like any other method so that the dispatch list doesn't
// need to be modified at any time. This is because modifying the dispatch list in any way would
// require writing an API with inverted control flow (i.e. `fn scoped(impl FnMut(Self) -> B) -> B`
// rather than `fn new() -> Self`) in order to ensure that non-root processes know what's up.
//
// (it would also make me want to refactor MpiOnDemand to make it more extensible, which is
//  something I really should not be spending my time doing given the fact that I have no reason
//  for using MPI other than Lammps)
//
// We are lucky that only a single lammps instance can exist at a time, because that
// allows `new` and `drop` to work more like stateful `init`, `deinit` routines.
impl InputNew {
    #[allow(unused_unsafe)]
    unsafe fn invoke_method(self, root: &impl mpi::Root, lammps: &mut Option<LammpsOwner>) -> OutputNew {
        let InputNew { argv } = self;
        assert!(lammps.is_none(), "BUG: new called before drop!");

        let argv_strs: Vec<&str> = argv.iter().map(|s| &s[..]).collect();

        let new_lammps = unsafe { LammpsOwner::with_mpi(root.as_communicator(), &argv_strs)? };
        *lammps = Some(new_lammps);
        Ok(())
    }
}

impl InputDrop {
    fn invoke_method(self, _: &impl mpi::Root, lammps: &mut Option<LammpsOwner>) -> OutputDrop {
        // NOTE: even if the previous MpiLammpsOwner was leaked, the instance Mutex would still be
        //       locked, making this mode of failure unreachable.
        assert!(lammps.is_some(), "BUG: drop called before new!");
        *lammps = None;
    }
}

/// This type exists to facilitate codegen.
pub(crate) type OutputNew = FailResult<()>;
/// This type exists to facilitate codegen.
pub(crate) type OutputDrop = ();
/// This type exists to facilitate codegen.
pub(crate) type OutputCommand = FailResult<()>;
/// This type exists to facilitate codegen.
pub(crate) type OutputGetNatoms = usize;
/// This type exists to facilitate codegen.
pub(crate) type OutputResetBox = FailResult<()>;
/// This type exists to facilitate codegen.
pub(crate) type OutputGatherAtomsI = FailResult<Vec<i64>>;
/// This type exists to facilitate codegen.
pub(crate) type OutputGatherAtomsF = FailResult<Vec<f64>>;
/// This type exists to facilitate codegen.
pub(crate) type OutputScatterAtomsI = FailResult<()>;
/// This type exists to facilitate codegen.
pub(crate) type OutputScatterAtomsF = FailResult<()>;
/// This type exists to facilitate codegen.
pub(crate) type OutputExtractCompute0d = FailResult<f64>;
/// This type exists to facilitate codegen.
pub(crate) type OutputExtractCompute1d = FailResult<Vec<f64>>;

//------------------------------------------------

impl Broadcast for Input {
    fn broadcast(root: &impl mpi::Root, input: Option<Input>) -> Input {
        // Have the root process tell all the other processes what variant it is.
        let method = input.as_ref().map(|input| match input {
            Input::New { .. } => Method::New,
            Input::Drop { .. } => Method::Drop,
            Input::Command { .. } => Method::Command,
            Input::GetNatoms { .. } => Method::GetNatoms,
            Input::ResetBox { .. } => Method::ResetBox,
            Input::GatherAtomsI { .. } => Method::GatherAtomsI,
            Input::GatherAtomsF { .. } => Method::GatherAtomsF,
            Input::ScatterAtomsI { .. } => Method::ScatterAtomsI,
            Input::ScatterAtomsF { .. } => Method::ScatterAtomsF,
            Input::ExtractCompute0d { .. } => Method::ExtractCompute0d,
            Input::ExtractCompute1d { .. } => Method::ExtractCompute1d,
        } as u32);
        let method = Broadcast::broadcast(root, method);
        let method = Method::from_int(method).unwrap();

        // Now that everybody knows what variant it is, transmit the data.
        macro_rules! gen_match {
            ($($Name:ident,)*) => {
                match method {
                    $(
                        // each variant of Input is just a wrapper around a `Broadcast`able struct,
                        // so broadcast that.
                        Method::$Name => {
                            let input = input.and_then(|x| match x {
                                Input::$Name(input) => Some(input),
                                _ => {
                                    // This should basically never come up, but if it does, it's
                                    // on a non-root process so the current value doesn't matter
                                    // anyways.
                                    assert!(!this_process_is_root(root), "BUG!");
                                    None
                                }
                            });
                            let input = Broadcast::broadcast(root, input);
                            Input::$Name(input)
                        }
                    )*
                }
            };
        }
        gen_match!{
            New, Drop,
            Command, GetNatoms, ResetBox,
            GatherAtomsI, GatherAtomsF,
            ScatterAtomsI, ScatterAtomsF,
            ExtractCompute0d, ExtractCompute1d,
        }
    }
}

impl Broadcast for ComputeStyle {
    fn broadcast(root: &impl mpi::Root, value: Option<Self>) -> Self {
        let id = Broadcast::broadcast(root, value.map(|x| x as u32));
        Self::from_int(id).unwrap()
    }
}

impl Broadcast for Skews {
    fn broadcast(root: &impl mpi::Root, value: Option<Self>) -> Self {
        let arr = value.map(|Skews { xy, xz, yz }| [xy, xz, yz]);
        let [xy, xz, yz] = Broadcast::broadcast(root, arr);
        Skews { xy, xz, yz }
    }
}

//------------------------------------------------

/// Represents the `MpiOnDemand` event loop body for Lammps.
#[derive(Debug)]
pub(crate) struct LammpsDispatch {
    // Perhaps surprisingly, the instance is stored here like some sort of singleton,
    // rather than explicitly on MpiLammpsOwner.
    //
    // If we did not do it this way, then the event loop for non-root processes
    // would have to maintain its own local variable for the instance, introducing yet
    // another place where New and Drop would have to be special-cased.
    //
    // FIXME: Unsurprisingly, I am uncertain whether this was a good idea.
    //        This design decision seems to have led to many other questionable design choices:
    //        - MpiLammpsOwner holding a LammpsOnDemand rather than a LammpsOwner
    //        - Arcs, Mutexes, Options, oh my!
    //        - Having to unsafe impl Send for CArgv
    //        - Making `MpiOnDemand` impl `Clone`, which is why `Dispatch` takes `&self`
    //        - Having to use interior mutability in our only implementation of `Dispatch`!
    //
    //        Unless there were other motivations for these decisions (which I can't recall),
    //        I imagine that we could obtain a much cleaner overall design if we did not put
    //        the LammpsOwner here.
    instance: Arc<Mutex<Option<LammpsOwner>>>,
}

impl LammpsDispatch {
    pub fn new() -> Self { LammpsDispatch { instance: Default::default() } }
}

impl DispatchMultiProcess for LammpsDispatch {
    type Input = Input;
    type Output = Output;

    fn dispatch(&self, root: &impl mpi::Root, input: Self::Input) -> Self::Output {
        let mut lammps = self.instance.try_lock().unwrap();
        let lammps = &mut lammps;

        // (NOTE: encapsulation of unsafety in this module is awkward/sloppy due to safe, private
        //        abstractions and codegen that wrap both safe and unsafe functions.
        //
        //        The unsafety in this block is nothing more than the contracts on individual
        //        methods on `LowLevelApi`. Because this method is only ever called from within
        //        implementations of those same trait methods, the contracts are already known
        //        to be upheld.)
        // (FIXME: we do need to iron up our defenses against bugs of the "processes falling out
        //         of sync" variety, as they would invalidate the above argument on non-root
        //         processes)
        unsafe {
            match input {
                Input::New(input) => Output::New(input.invoke_method(root, lammps)),
                Input::Drop(input) => Output::Drop(input.invoke_method(root, lammps)),
                Input::Command(input) => Output::Command(input.invoke_method(root, lammps)),
                Input::GetNatoms(input) => Output::GetNatoms(input.invoke_method(root, lammps)),
                Input::ResetBox(input) => Output::ResetBox(input.invoke_method(root, lammps)),
                Input::GatherAtomsI(input) => Output::GatherAtomsI(input.invoke_method(root, lammps)),
                Input::GatherAtomsF(input) => Output::GatherAtomsF(input.invoke_method(root, lammps)),
                Input::ScatterAtomsI(input) => Output::ScatterAtomsI(input.invoke_method(root, lammps)),
                Input::ScatterAtomsF(input) => Output::ScatterAtomsF(input.invoke_method(root, lammps)),
                Input::ExtractCompute0d(input) => Output::ExtractCompute0d(input.invoke_method(root, lammps)),
                Input::ExtractCompute1d(input) => Output::ExtractCompute1d(input.invoke_method(root, lammps)),
            }
        }
    }
}
