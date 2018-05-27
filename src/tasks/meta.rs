//! Site metadata
//!
//! High-level code carts site metadata around in the form of `HLists` of
//! `Rc<[T]>` (or similar, like `Option<Rc<[T]>>`). The use of `Rc` eliminates
//! concern about ownership (it is *distinctly un-fun* to turn an `Option<Vec<T>>`
//! into an `Option<&[T]`), and allows this module to get away with a very small
//! number of extension traits. (where a reasonably ergonomic API that can handle
//! differences in ownership would require at least half a dozen)

use ::std::fmt;

macro_rules! derive_newtype_display {
    ($Type:ident) => {
        impl fmt::Display for $Type {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
#[derive(Serialize, Deserialize)]
pub struct Mass(pub f64);
derive_newtype_display!{ Mass }

pub use ::rsp2_structure::Element;



pub mod prelude {
    pub use super::MetaSift;
    pub use super::MetaPick;
}

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
