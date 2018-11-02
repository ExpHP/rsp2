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

//! Allows one to write code which only uses MPI on demand, allowing the bulk
//! of an MPI codebase to be executed only on the root process.
//!
//! This is useful for wrapping a library that uses MPI in an application that
//! does not otherwise need MPI, preventing the MPI "multi-process, single-code"
//! model from infecting large portions of code.
//!
//! ...there are, however, massive downsides. Even though only the root is running most of
//! the code, the other processes still need to know which MPI functions are going to be used.
//! To make this work, the handler function must be written largely as a state machine.
//!
//! # Limitations and suspected footguns
//!
//! This abstraction **does not compose with itself very well.**  The intended use-case is to
//! install a single scoped instance near the beginning of main, for using MPI in a single library.
//!
//! If you need to support more than one library, you can probably write a Dispatch that dispatches
//! to multiple other Dispatches...? I think a Dispatch can even make a nested call to `install`?
//! I dunno.  Haven't tried it, don't need it, and I would probably try to come up with
//! a more scalable design if I *did* need it.

use ::mpi;
use ::std::sync::Arc;
use ::slice_of_array::prelude::*;

/// A multi-process entry point for MpiOnDemand.
///
/// # MPI
///
/// An instance of one of these objects exists on all processes.
/// The `install` function will allow you to enter single-process mode.
pub trait DispatchMultiProcess {
    type Input: Broadcast;
    type Output;

    // NOTE: `&self` rather than `&mut self` because MpiOnDemand impls Clone.
    //       (honestly though I wish this were not the case; see the big fixme in LammpsDispatch)
    fn dispatch(&self, root: &impl mpi::Root, input: Self::Input) -> Self::Output;

}

/// Allows code running on a single, root process to invoke an MPI function on all processes
/// on demand.
///
/// See the module-level documentation for more details.
///
/// It is impossible to obtain one of these on a non-root process.
///
/// It must not be leaked outside the closure given to `install`.
///
/// # Shared state
///
/// An `MpiOnDemand` contains nothing more than a handle to the `Dispatch`.
/// **If it is cloned, all clones will have the same `Dispatch`.**
/// This matters in the case that a `Dispatch` uses interior mutability
/// (mind: *this is in fact true for the only currently existing `Dispatch`!*),
//
// FIXME: The Arc is so that the closure for `install` can receive something that is `'static`.
//        This requirement comes from the sole intended use case; I did not want to have to add a
//        lifetime yet to PotentialBuilder (something which appears all over `rsp2_tasks`) just to
//        support this one feature that might not even pan out.
//
// NOTE: This used to be parameterized over the communicator (or something
//       that implements mpi::Root) in order to support UserCommunicators.
//
//       I gave up on that because it is simply too hard:
//
//       - The communicator (or Process) must be stored in `MpiOnDemand`
//       - mpi::Communicator is not object-safe
//       - mpi::Root is not object-safe, even if you constrain AsCommunicator::Out
//       - The existing Send+Sync impls on mpi types are arbitrary, unsound, and platform-dependent.
//         (https://github.com/bsteinb/rsmpi/issues/12)
//       - It is impossible to obtain a `Process` that is `'static`. As a result, lifetime
//         params are forced to appear all the way up through `rsp2_tasks::PotentialBuilder`.
//
#[derive(Debug)]
pub struct MpiOnDemand<D>(Arc<MpiOnDemandInner<D>>);

impl<D> Clone for MpiOnDemand<D> {
    fn clone(&self) -> Self { MpiOnDemand(self.0.clone()) }
}

#[derive(Debug, Clone)]
pub struct MpiOnDemandInner<D> {
    dispatch: D,
}

impl<D: DispatchMultiProcess> MpiOnDemand<D> {
    /// Run the provided closure on a single process, with all multi-process code factored out
    /// into the provided `Dispatch`.
    ///
    /// The closure will receive an object which allows one to invoke the `Dispatch` on demand.
    /// This object must not be leaked.
    ///
    /// # MPI
    ///
    /// This method is called on all processes.  The closure is called on the root process,
    /// while the others enter an event loop (mostly blocking, waiting for `invoke` to be called).
    /// When the closure exits, execution resumes on all processes.
    ///
    /// **Important:** Currently, if a process panics at a bad time, the other processes may
    /// get deadlocked on a blocking operation. Nothing is done about this by default.
    /// See the helper method `with_mpi_abort_on_unwind` for a possible solution.
    ///
    /// # Panics
    ///
    /// Panics after the function returns if it is detected that the `MpiOnDemand`
    /// has been leaked.
    pub fn install<R>(
        dispatch: D,
        func: impl FnOnce(MpiOnDemand<D>) -> R,
    ) -> Option<R> {
        let on_demand = MpiOnDemandInner { dispatch };

        // a note to future me (because he's an idiot):
        //
        // The continuation on the next line is entirely incidental and unrelated to the reason why
        // this function takes a continuation. (and that reason is a very good one: to delimit
        // the scope of single-process mode.)
        with_default_root(|root| {
            if this_process_is_root(&root) {
                let on_demand = MpiOnDemand(Arc::new(on_demand));
                let out = func(on_demand.clone());

                Arc::try_unwrap(on_demand.0).ok()
                    .expect("Detected leak of `MpiOnDemand` value!")
                    .root_finish(&root);

                Some(out)
            } else {
                on_demand.non_root_event_loop(&root);
                None
            }
        })
    }

    /// Call the multi-process entry point.
    ///
    /// # MPI
    ///
    /// Because `MpiOnDemand` is only accessible on the root process, it should go without
    /// saying that this method only needs to be called on the root.
    ///
    /// The multi-process entry point will be invoked on all processes simultaneously,
    /// but the the return value will be ignored on any non-root process.
    pub fn invoke(&self, arg: D::Input) -> D::Output {
        with_default_root(|root| self.0.root_dispatch(&root, arg))
    }

    /// Place the non-root processes into a low activity state for the duration of a closure.
    ///
    /// Normally, the non-root processes for `MpiOnDemand` spend most of their time in an
    /// `MPI_Bcast`, which is typically a busy wait with `~100%` CPU usage for performance reasons.
    /// `eco_mode` puts them into a much less active state so that they do not heavily compete
    /// against other threads and processes when they aren't being used for anything.
    ///
    /// # Potential for deadlocks
    ///
    /// Calling any method of `MpiOnDemand` from within the closure will result in a communication
    /// failure between the root process and the others.
    ///
    /// In an ideal world, this method would take `&mut self` to make such mistakes impossible.
    /// However, it currently does not because `MpiOnDemand` impls `Clone` (making the `&mut` easily
    /// circumvented), and because `rsp2_tasks::PotentialBuilder` (the only place where this is
    /// ultimately used) still exposes a thoroughly `&self`-based API.
    pub fn eco_mode<B>(&self, cont: impl FnOnce() -> B) -> B {
        with_default_root(|root| self.0.root_eco_mode(&root, cont))
    }
}

/// Helper to call `MPI_ABORT` if a panic occurs inside the continuation,
/// *after* allowing the panic implementation to unwind back out.
///
/// This will be completely ineffective if the panic implementation does not unwind.
pub fn with_mpi_abort_on_unwind<R>(func: impl ::std::panic::UnwindSafe + FnOnce() -> R) -> R {
    use ::mpi::{AsCommunicator, Communicator};

    with_default_root(|root| {
        let res = ::std::panic::catch_unwind(func);
        match res {
            Ok(r) => return r,
            Err(_payload) => {
                // we won't need to worry about printing a message, under the assumption
                // that the panic hook already did so before beginning to unwind.
                root.as_communicator().abort(1);
            },
        }
    })
}

// Provides the default `mpi::Root`.
//
// This exists because I had to give up on making the final product generic over Communicators.
//
// It is returned continuation-style because it is impossible to construct one that is `'static`.
fn with_default_root<R>(continuation: impl FnOnce(mpi::Process<'_, mpi::SystemCommunicator>) -> R) -> R {
    use ::mpi::Communicator;

    let world = mpi::SystemCommunicator::world();
    let root = world.process_at_rank(0);
    continuation(root)
}

impl<D: DispatchMultiProcess> MpiOnDemandInner<D> {
    fn root_dispatch(&self, root: &impl mpi::Root, arg: D::Input) -> D::Output {
        assert!(this_process_is_root(root));
        assert_eq!(Broadcast::broadcast(root, Some(EventType::Dispatch)), EventType::Dispatch);

        let arg = Broadcast::broadcast(root, Some(arg));
        self.dispatch.dispatch(root, arg)
    }

    fn non_root_dispatch(&self, root: &impl mpi::Root) {
        assert!(!this_process_is_root(root));

        let arg = Broadcast::broadcast(root, None);
        let _ = self.dispatch.dispatch(root, arg);
    }

    // ------------

    const ECO_MODE_FINISH: i32 = 1;

    fn root_eco_mode<B>(&self, root: &impl mpi::Root, cont: impl FnOnce() -> B) -> B {
        use ::mpi::{Destination, Communicator};

        // Inform the others.
        assert!(this_process_is_root(root));
        assert_eq!(Broadcast::broadcast(root, Some(EventType::EnterEcoMode)), EventType::EnterEcoMode);

        // NOTE: in the case where `cont` panics, we assume MPI will detect it and
        //       destroy the other processes before long
        let out = cont();

        // Make the others resume busy-waiting.
        // We use Send so that the others can use Iprobe.
        // (there is also Ibcast but it seems overkill)
        let world = root.as_communicator();
        for rank in 0..world.size() {
            if rank != world.rank() {
                world.process_at_rank(rank).send(&Self::ECO_MODE_FINISH);
            }
        }
        out
    }

    fn non_root_eco_mode(&self, root: &impl mpi::Root) {
        use ::mpi::{Source, Communicator};

        assert!(!this_process_is_root(root));

        // HACK: force into type Process because `mpi::Root` doesn't imply `mpi::Source`.
        let root = root.as_communicator().process_at_rank(root.root_rank());

        // Wait for the root process to signal that the closure has finished.
        while root.immediate_probe().is_none() {
            // NOTE: Because the root process communicates with the others one-by-one,
            //       the total time taken to resume N processes may be as large as `N * interval`.
            let interval = ::std::time::Duration::from_secs(1);
            ::std::thread::sleep(interval);
        }

        match root.receive().0 {
            Self::ECO_MODE_FINISH => {},
            n => panic!("unexpected value {} on exiting eco mode", n)
        }
    }

    // ------------

    fn root_finish(self, root: &impl mpi::Root) {
        assert!(this_process_is_root(root));
        assert_eq!(Broadcast::broadcast(root, Some(EventType::Finish)), EventType::Finish);
    }

    // ------------

    fn non_root_event_loop(self, root: &impl mpi::Root) {
        // FIXME: what about error handling?
        loop {
            match Broadcast::broadcast(root, None::<EventType>) {
                EventType::Dispatch => self.non_root_dispatch(root),
                EventType::EnterEcoMode => self.non_root_eco_mode(root),
                EventType::Finish => break,
            }
        }
    }
}

c_enums!{
    [] enum EventType {
        Dispatch = 0,
        EnterEcoMode = 1,
        Finish = 2,
    }
}

//---------------------------------------------------------------

/// Helper trait to broadcast data from the root process to all processes,
/// including vectors of unknown length, and types with no default.
pub trait Broadcast: Sized {
    /// Broadcast a value from the root to other threads.
    ///
    /// The value of `value` is ignored on non-root threads, and must be `Some` on the
    /// root thread.
    ///
    // NOTE: Even though many impls defer to an implementation with a `&mut Self` signature,
    //       the trait signature is `Option<Self> -> Self` because that is more versatile,
    //       and neither signature is capable of implementing the other without an additional
    //       bound like Default.
    fn broadcast(root: &impl mpi::Root, value: Option<Self>) -> Self;
}

// impl for a type that implements `mpi::BufferMut + Default`
macro_rules! impl_broadcast_for_buffer_mut {
    ($($T:ident)*) => {$(
        impl Broadcast for $T {
            fn broadcast(root: &impl mpi::Root, value: Option<$T>) -> $T
            { broadcast_into_buffer_mut(root, value) }
        }
    )*};
}
impl_broadcast_for_buffer_mut! {
    i8 i16 i32 i64 isize
    u8 u16 u32 u64 usize
    f32 f64 bool
}

// impl for Vec<T> where T implements `mpi::Equivalence + Copy + Default`
//
// This differs from the `mpi` crate's own `Root::broadcast_into::<[T]>` in that
// it can resize the vectors to match.
macro_rules! impl_broadcast_for_vec_equivalence {
    ($(Vec<$T:ident>)*) => {$(
        impl Broadcast for Vec<$T> {
            fn broadcast(root: &impl mpi::Root, buf: Option<Vec<$T>>) -> Vec<$T>
            { broadcast_vec(root, buf) }
        }
    )*};
}

impl_broadcast_for_vec_equivalence! {
    Vec<i64> Vec<f64>
}

impl Broadcast for String {
    fn broadcast(
        root: &impl mpi::Root,
        buf: Option<String>,
    ) -> String {
        let bytes = broadcast_vec(root, buf.map(|s| s.into_bytes()));
        String::from_utf8(bytes).unwrap()
    }
}

impl Broadcast for Vec<String> {
    fn broadcast(
        root: &impl mpi::Root,
        buf: Option<Vec<String>>,
    ) -> Vec<String> {
        broadcast_via_mut_ref(root, buf, |root, buf| {
            let mut count = buf.len();
            root.broadcast_into(&mut count);

            buf.resize(count, String::new());

            for s in &mut buf[..] {
                broadcast_into_via_option(root, s, Broadcast::broadcast);
            }
        })
    }
}

// Helper that adapts existing functions with a signature like `Root::broadcast_into`
// into a Broadcast impl, given the existence of a `Default` impl.
fn broadcast_via_mut_ref<T, R>(
    root: &R,
    value: Option<T>,
    broadcast_into: impl FnOnce(&R, &mut T),
) -> T
where
    R: mpi::AsCommunicator + mpi::Root,
    T: Default,
{
    if this_process_is_root(root) && value.is_none() {
        panic!("root did not provide value to broadcast");
    }
    let mut buf = value.unwrap_or_else(Default::default);
    broadcast_into(root, &mut buf);
    buf
}

// Reverse of `broadcast_via_mut_ref`.
fn broadcast_into_via_option<T, R>(
    root: &R,
    buf: &mut T,
    broadcast: impl FnOnce(&R, Option<T>) -> T,
)
where
    R: mpi::AsCommunicator + mpi::Root,
    T: Default,
{
    let value = ::std::mem::replace(buf, Default::default());
    let value = broadcast(root, Some(value));
    *buf = value;
}

// For creating `Broadcast` impls from types that implement BufferMut.
fn broadcast_into_buffer_mut<T>(
    root: &impl mpi::Root,
    value: Option<T>,
) -> T
where T: Default + mpi::BufferMut,
{ broadcast_via_mut_ref(root, value, |root, buf| root.broadcast_into(buf)) }

fn broadcast_vec<T: mpi::Equivalence + Copy + Default>(
    root: &impl mpi::Root,
    buf: Option<Vec<T>>,
) -> Vec<T> {
    broadcast_via_mut_ref(root, buf, |root, buf| {
        let mut size = buf.len();
        root.broadcast_into(&mut size);

        if !this_process_is_root(root) {
            buf.resize(size, T::default());
        }
        root.broadcast_into(&mut buf[..]);
    })
}

impl Broadcast for [f64; 3] {
    fn broadcast(root: &impl mpi::Root, value: Option<[f64; 3]>) -> [f64; 3] {
        broadcast_via_mut_ref(root, value, |root, buf| {
            root.broadcast_into(&mut buf[..]);
        })
    }
}

impl<T> Broadcast for Vec<[T; 3]>
where
    Vec<T>: Broadcast,
    T: Clone,
{
    fn broadcast(root: &impl mpi::Root, value: Option<Vec<[T; 3]>>) -> Vec<[T; 3]> {
        let flat = Broadcast::broadcast(root, value.map(|vec| vec.flat().to_vec()));
        flat.nest().to_vec()
    }
}

macro_rules! impl_broadcast_for_c_enum {
    ($($T:ident)*) => {$(
        impl Broadcast for $T {
            fn broadcast(root: &impl mpi::Root, buf: Option<$T>) -> $T {
                $T::from_int(Broadcast::broadcast(root, buf.map(|x| x as _))).unwrap()
            }
        }
    )*};
}

impl_broadcast_for_c_enum! { EventType }

//---------------------------------------------------------------

pub fn this_process_is_root(root: &impl mpi::Root) -> bool
{ mpi::Communicator::rank(root.as_communicator()) == root.root_rank() }
