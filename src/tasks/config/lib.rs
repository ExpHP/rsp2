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

#![allow(non_snake_case)]

//! Crate where serde_yaml code for the 'tasks' crate is monomorphized,
//! because this is a huge compile time sink.
//!
//! The functions here also make use of serde_ignored to catch typos in the config.

// NOTE: Please make sure to use the YamlRead trait!
//       DO NOT USE serde_yaml::from_{reader,value,etc.} OUTSIDE THIS CRATE
//       or else you defeat the entire reason for its existence.

// (NOTE: I can't enforce this through the type system without completely destroying
//        the ergonomics of these types. Just Ctrl+Shift+F the workspace for "serde_yaml"
//        if compile times seem suspiciously off...)

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate log;
#[macro_use]
extern crate failure;
#[macro_use]
extern crate rsp2_util_macros;
#[macro_use]
extern crate rsp2_config_utils;

pub use config::*;
mod config;

mod validation;

mod option_aliases {
    /// Alias used for `Option<T>` to indicate that this field has a default which is implemented
    /// outside of this module. (e.g. in the implementation of `Default` or `new` for a builder
    /// somewhere)
    pub type OrDefault<T> = Option<T>;

    /// Alias used for `Option<T>` to indicate that omitting this field has special meaning.
    pub type Nullable<T> = Option<T>;

    /// Newtype around `Option<T>` for fields that are guaranteed to be `Some` after the
    /// config is validated. Used for e.g. the new location of a deprecated field so that
    /// it can fall back to reading from the old location.
    #[derive(Serialize, Deserialize)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Filled<T>(pub(crate) Option<T>);
    impl<T> Filled<T> {
        pub(crate) fn default() -> Self { Filled(None) }

        pub fn into_inner(self) -> T { self.0.unwrap() }
        pub fn as_ref(&self) -> &T { self.0.as_ref().unwrap() }
        pub fn as_mut(&mut self) -> &mut T { self.0.as_mut().unwrap() }
    }

    impl<T> From<T> for Filled<T> {
        fn from(x: T) -> Self { Filled(Some(x)) }
    }
}
