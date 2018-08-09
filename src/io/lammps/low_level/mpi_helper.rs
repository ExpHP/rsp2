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
                    .finish_from_root(&root);

                Some(out)
            } else {
                non_root_event_loop(&root, on_demand);
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
        with_default_root(|root| self.0.invoke_from_root(&root, arg))
    }
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
    fn invoke_from_root(&self, root: &impl mpi::Root, arg: D::Input) -> D::Output {
        assert!(this_process_is_root(root), "BUG!");
        assert!(Broadcast::broadcast(root, Some(true)), "BUG!");

        let arg = Broadcast::broadcast(root, Some(arg));
        self.dispatch.dispatch(root, arg)
    }

    fn invoke_from_non_root(&self, root: &impl mpi::Root) -> KeepGoing {
        assert!(!this_process_is_root(root), "BUG!");

        let keep_going = Broadcast::broadcast(root, None::<bool>);
        if keep_going {
            let arg = Broadcast::broadcast(root, None);
            let _ = self.dispatch.dispatch(root, arg);
        }
        KeepGoing(keep_going)
    }

    fn finish_from_root(self, root: &impl mpi::Root) {
        assert!(this_process_is_root(root), "BUG!");

        // Make the other processes exit the event loop.
        assert!(!Broadcast::broadcast(root, Some(false)), "BUG!");
    }
}

struct KeepGoing(bool);

// FIXME: what about error handling?
fn non_root_event_loop<D: DispatchMultiProcess>(
    root: &impl mpi::Root,
    on_demand: MpiOnDemandInner<D>,
) {
    while let KeepGoing(true) = on_demand.invoke_from_non_root(root) { }
}

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

pub fn this_process_is_root(root: &impl mpi::Root) -> bool
{ mpi::Communicator::rank(root.as_communicator()) == root.root_rank() }
