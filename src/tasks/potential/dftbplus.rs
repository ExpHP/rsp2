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

// (note: GPL necessary due to usage of things in super::)

//! All usage of the public API presented by `rsp2_dftbplus` is encapsulated here.

use super::{DynCloneDetail, PotentialBuilder, DiffFn, DispFn, CommonMeta, BondDiffFn};
use crate::FailResult;
#[allow(unused)] // rustc bug
use crate::meta::{self, prelude::*};
#[allow(unused)] // rustc bug
use rsp2_soa_ops::{Part, Partition};
use rsp2_structure::{Coords};
use rsp2_array_types::{V3};
use rsp2_tasks_config as cfg;
use crate::cmd::trial::TrialDir;

use rsp2_dftbplus as wrapper;

/// A bundle of everything we need to initialize a DFTB+ API object.
///
/// It is nothing more than a bundle of configuration, and can be freely
/// sent across threads.
#[derive(Debug, Clone)]
pub(crate) struct Builder {
    inner: wrapper::Builder,
}

#[allow(unused)] // compile-time test
fn assert_builder_send_sync() {
    fn assert_send_sync<S: Send + Sync>() {}

    assert_send_sync::<Builder>();
}

impl Builder {
    pub(crate) fn new(
        trial_dir: Option<&TrialDir>,
        cfg: &cfg::PotentialDftbPlus,
    ) -> FailResult<Self> {
        let hsd = cfg.hsd.parse::<wrapper::Hsd>()?;
        let mut inner = wrapper::Builder::from_hsd(&hsd);
        if let Some(trial_dir) = trial_dir {
            inner.append_log(trial_dir.as_path().join("dftb+.log"));
        }

        Ok(Builder { inner })
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// FIXME: Workaround for https://github.com/dftbplus/dftbplus/issues/247 .
//        Basically: avoid calling dftbp_final, ever.
//
//        Okay only because we know that RSP2 never tries to compute structures where
//        anything besides the coords or lattice have changed. (i.e. nothing ever changes
//        that would require a new instance of DFTB+)
//        (unless you try to do a supercell, in which case this may do bad things.
//         Hope for a quick fix upstream!)
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mod cache {
    use super::*;

    lazy_static!{
        static ref CACHED_INSTANCE: std::sync::Mutex<Option<FakeSendSync<wrapper::DftbPlus>>> = {
            std::sync::Mutex::new(None)
        };
    }

    pub fn get_or_create<E>(f: impl FnOnce() -> Result<wrapper::DftbPlus, E>) -> Result<Cached, E> {
        let mut guard = CACHED_INSTANCE.try_lock().expect("cannot use DFTB+ in parallel");
        if guard.is_none() {
            let dftb = f()?;
            // FIXME FIXME FIXME
            // probably not correct, but all of this can be removed as soon
            // as https://github.com/dftbplus/dftbplus/issues/247 is fixed
            let dftb = unsafe { FakeSendSync::new_unchecked(dftb) };
            *guard = Some(dftb);
        }
        Ok(Cached(guard))
    }

    // A MutexGuard<Option<T>> that is known to be Some
    pub struct Cached(std::sync::MutexGuard<'static, Option<FakeSendSync<wrapper::DftbPlus>>>);

    impl std::ops::Deref for Cached {
        type Target = wrapper::DftbPlus;

        fn deref(&self) -> &Self::Target
        { &**self.0.as_ref().unwrap() }
    }

    impl std::ops::DerefMut for Cached {
        fn deref_mut(&mut self) -> &mut Self::Target
        { &mut **self.0.as_mut().unwrap() }
    }

    use fake_send_sync::FakeSendSync;
    mod fake_send_sync {
        pub struct FakeSendSync<T>(T);

        unsafe impl<T> Send for FakeSendSync<T> { }
        unsafe impl<T> Sync for FakeSendSync<T> { }

        impl<T> FakeSendSync<T> {
            pub unsafe fn new_unchecked(x: T) -> Self { FakeSendSync(x) }
        }

        impl<T> std::ops::Deref for FakeSendSync<T> {
            type Target = T;

            fn deref(&self) -> &Self::Target { &self.0 }
        }

        impl<T> std::ops::DerefMut for FakeSendSync<T> {
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
        }
    }
}

impl PotentialBuilder<CommonMeta> for Builder {
    fn initialize_bond_diff_fn(&self, _: &Coords, _: CommonMeta) -> FailResult<Option<Box<dyn BondDiffFn<CommonMeta>>>>
    { Ok(None) }

    /// Initialize DFTB+ to make a DiffFn.
    ///
    /// This keeps the DFTB+ instance between calls to save time.
    ///
    /// Some data may be pre-allocated or precomputed based on the input structure,
    /// so the resulting DiffFn may not support arbitrary structures as input.
    fn initialize_diff_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DiffFn<CommonMeta>>>
    {
        let elements: meta::SiteElements = meta.pick();

        let mut dftb_builder = self.inner.clone();
        let dftb = cache::get_or_create(|| {
            dftb_builder
                .initial_coords(coords)
                .elements(&elements)
                .build()
        })?;

        // a DiffFn 'lambda' whose type will be erased
        struct MyDiffFn(cache::Cached);
        //struct MyDiffFn(wrapper::DftbPlus);
        impl DiffFn<CommonMeta> for MyDiffFn {
            fn compute(&mut self, coords: &Coords, meta: CommonMeta) -> FailResult<(f64, Vec<V3>)> {
                let dftb = &mut self.0;

                let elements: meta::SiteElements = meta.pick();
                if &elements[..] != dftb.elements() {
                    bail!{ "\
                        Detected change in site elements! This is not supported by \
                        the dftb+ potential.\
                    "};
                }

                dftb.set_coords(coords)?;

                let value = dftb.compute_value()?;
                let grad = dftb.compute_grad()?;
                Ok((value, grad))
            }
        }

        Ok(Box::new(MyDiffFn(dftb)) as Box<_>)
    }

    fn initialize_disp_fn(&self, coords: &Coords, meta: CommonMeta) -> FailResult<Box<dyn DispFn>>
    { self._default_initialize_disp_fn(coords, meta) }
}

impl_dyn_clone_detail!{
    impl[] DynCloneDetail<CommonMeta> for Builder { ... }
}
