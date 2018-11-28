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

//! Site metadata
//!
//! High-level code carts site metadata around in the form of `HLists` of
//! `Rc<[T]>` (or similar, like `Option<Rc<[T]>>`). The use of `Rc` eliminates
//! concern about ownership (it is *distinctly un-fun* to turn an `Option<Vec<T>>`
//! into an `Option<&[T]`), and allows this module to get away with a very small
//! number of extension traits. (where a reasonably ergonomic API that can handle
//! differences in ownership would require at least half a dozen)

use std::fmt;
use std::rc::Rc;

macro_rules! derive_newtype_display {
    ($Type:ident) => {
        impl fmt::Display for $Type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

//----------------------------------------------------------------------------------
// Types that appear inside metadata slices to serve as newtypes.

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
#[derive(Serialize, Deserialize)]
pub struct Mass(pub f64);
derive_newtype_display!{ Mass }

pub use rsp2_structure::Element;

newtype_index!{
    #[derive(Serialize, Deserialize)]
    Layer
}

//----------------------------------------------------------------------------------

// Names intended for use in HList types.
// The names here deliberately conflict with other names in the crate;
// these are not intended to be imported, but rather written as `meta::Masses`, etc.
pub type SiteMasses = Rc<[Mass]>;
pub type SiteElements = Rc<[Element]>;
pub type SiteLayers = Rc<[Layer]>;
pub type LayerScMatrices = Rc<[crate::math::bands::ScMatrix]>;
pub type FracBonds = Rc<rsp2_structure::bonds::FracBonds>;
pub type CartBonds = Rc<rsp2_structure::bonds::CartBonds>;

//----------------------------------------------------------------------------------

pub mod prelude {
    pub use super::MetaSift;
    pub use super::MetaPick;
    pub use super::MetaSendable;
}

//----------------------------------------------------------------------------------

/// Get a subset of a metadata list by type, in any order.
///
/// Equivalent to `list.clone().sculpt().0`.
///
/// Intended for use on Metadata lists, which are (relatively) cheap to clone.
pub trait MetaSift<Targets, Indices> {
    fn sift(&self) -> Targets;
}

impl<List, Targets, Indices> MetaSift<Targets, Indices> for List
where
    List: Clone,
    List: ::frunk::hlist::Sculptor<Targets, Indices>,
{
    fn sift(&self) -> Targets
    { self.clone().sculpt().0 }
}

/// Get a single item of a metadata list by type.
///
/// Equivalent to `list.get().clone()`.
///
/// Intended for use on Metadata lists, which are (relatively) cheap to clone.
pub trait MetaPick<Target, Index> {
    fn pick(&self) -> Target;
}

impl<List, Target, Index> MetaPick<Target, Index> for List
where
    Self: ::frunk::hlist::Selector<Target, Index>,
    Target: Clone,
{
    fn pick(&self) -> Target
    { self.get().clone() }
}

//----------------------------------------------------------------------------------

/// Workaround to use metadata where thread safety is required.
///
/// Basically, metadata is not threadsafe due to heavy use of Rc.
/// This makes a sendable function that produces a copy of Self
/// each time it is called.
pub trait MetaSendable: Sized + Clone {
    fn sendable<'a>(&'a self) -> Box<Fn() -> Self + Send + Sync + 'a>;
}

impl<T: Clone + Sync> MetaSendable for ::std::rc::Rc<[T]> {
    fn sendable<'a>(&'a self) -> Box<Fn() -> Self + Send + Sync + 'a> {
        let send = &self[..];
        Box::new(move || send.into())
    }
}

impl MetaSendable for FracBonds {
    fn sendable<'a>(&'a self) -> Box<Fn() -> Self + Send + Sync + 'a> {
        let send = &**self;
        Box::new(move || Rc::new(send.clone()))
    }
}

impl<V: MetaSendable> MetaSendable for Option<V> {
    fn sendable<'a>(&'a self) -> Box<Fn() -> Self + Send + Sync + 'a> {
        let get = self.as_ref().map(|x| x.sendable());
        Box::new(move || get.as_ref().map(|f| f()))
    }
}

impl<A, B> MetaSendable for ::frunk::HCons<A, B>
where
    A: MetaSendable,
    B: MetaSendable,
{
    fn sendable<'a>(&'a self) -> Box<Fn() -> Self + Send + Sync + 'a> {
        let get_a = self.head.sendable();
        let get_b = self.tail.sendable();
        Box::new(move || hlist![get_a(), ...get_b()])
    }
}

impl MetaSendable for ::frunk::HNil {
    fn sendable<'a>(&'a self) -> Box<Fn() -> Self + Send + Sync + 'a> {
        Box::new(|| ::frunk::HNil)
    }
}

//----------------------------------------------------------------------------------
