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

#[cfg(feature = "frunk")]
use frunk::{HCons, HNil};

use perm_vec as imp;

pub use imp::Perm;
pub use imp::InvalidPermutationError;

// NOTE: We wrap the trait from perm_vec in order to add impls on types like Option and HCons.

/// Trait for applying a permutation operation.
///
/// `rsp2` uses permutations for a wide variety of purposes, though most
/// frequently they represent a permutation of atoms.
///
/// Impls of `Permute` do not always necessarily apply the permutation directly to
/// vectors contained in the type.  For instance, data in a sparse format will likely
/// use the permutation to transform stored indices.  It helps to consider indices
/// as having distinct types (e.g. site indices, image indices; or more specific things
/// like "indices of sites when sorted by x"); in this model, `Perm` can be thought of
/// as being parametrized over two index types, where `Perm<Src, Dest>` transforms
/// data indexed by type `Src` into data indexed by type `Dest`.
///
/// # Laws
///
/// All implementations of `Permute` must satisfy the following properties,
/// which give `Permute::permuted_by` the qualities of a group action.
/// (whose group operator is, incidentally, also `Permute::permuted_by`!)
///
/// * **Identity:**
///   ```text
///   data.permuted_by(Perm::eye(data.len())) == data
///   ```
/// * **Compatibility:**
///   ```text
///   data.permuted_by(a).permuted_by(b) == data.permuted_by(a.permuted_by(b))
///   ```
///
/// When envisioning `Perm` as generic over `Src` and `Dest` types, it could
/// perhaps be said that `Perm`s are the morphisms of a category. (brushing
/// aside issues of mismatched length)
pub trait Permute: Sized {
    // awkward name, but it makes it makes two things clear
    // beyond a shadow of a doubt:
    // - The receiver gets permuted, not the argument.
    //   (relevant when Self is Perm)
    // - The permutation is not in-place.
    fn permuted_by(self, perm: &Perm) -> Self;
}

impl<T> Permute for Vec<T> {
    fn permuted_by(self, perm: &Perm) -> Vec<T>
    { imp::Permute::permuted_by(self, perm) }
}

impl Permute for Perm {
    fn permuted_by(self, other: &Perm) -> Perm
    { self.then(other) }
}

// combinators
#[cfg(feature = "frunk")]
impl Permute for HNil {
    fn permuted_by(self, _: &Perm) -> HNil
    { HNil }
}

#[cfg(feature = "frunk")]
impl<A, B> Permute for HCons<A, B>
where
    A: Permute,
    B: Permute,
{
    fn permuted_by(self, perm: &Perm) -> HCons<A, B>
    { HCons {
        head: self.head.permuted_by(perm),
        tail: self.tail.permuted_by(perm),
    }}
}

impl<A> Permute for Option<A>
where
    A: Permute,
{
    fn permuted_by(self, perm: &Perm) -> Option<A>
    { self.map(|x| x.permuted_by(perm)) }
}

// rsp2-tasks needs this
impl<T: Clone> Permute for std::rc::Rc<[T]> {
    fn permuted_by(self, perm: &Perm) -> Self
    { imp::Permute::permuted_by(self, perm) }
}

impl<T: Permute + Clone> Permute for std::rc::Rc<T> {
    fn permuted_by(self, perm: &Perm) -> Self {
        Box::new((*self).clone().permuted_by(perm)).into()
    }
}
