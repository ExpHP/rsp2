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

use std::io::Read;
use failure::Error;

pub use monomorphize::YamlRead;
#[macro_use]
mod monomorphize {
    use super::*;

    /// Provides an alternative to serde_yaml::from_reader where all of the
    /// expensive codegen has already been performed in this crate.
    pub trait YamlRead: for <'de> serde::Deserialize<'de> {
        fn from_reader(mut r: impl Read) -> Result<Self, Error>
        { YamlRead::from_dyn_reader(&mut r) }

        fn from_dyn_reader(r: &mut dyn Read) -> Result<Self, Error> {
            // serde_ignored needs a Deserializer.
            // unlike serde_json, serde_yaml doesn't seem to expose a Deserializer that is
            // directly constructable from a Read... but it does impl Deserialize for Value.
            //
            // However, on top of that, deserializing a Value through serde_ignored makes
            // one lose all of the detail from the error messages. So...
            //
            // First, parse to a form that we can read from multiple times.
            let mut s = String::new();
            r.read_to_string(&mut s)?;

            // try deserializing from Value, printing warnings on unused keys.
            // (if value_from_dyn_reader fails, that error should be fine)
            let value = value_from_str(&s)?;

            match Self::__serde_ignored__from_value(value) {
                Ok(out) => Ok(out),
                Err(_) => {
                    // That error message was surely garbage. Let's re-parse again
                    // from the string, without serde_ignored:
                    Self::__serde_yaml__from_str(&s)?;
                    unreachable!();
                }
            }
        }

        // trait-provided function definitions seem to be lazily monomorphized, so we
        // must put the meat of what we need monomorphized directly into the impls
        #[doc(hidden)]
        fn __serde_ignored__from_value(value: serde_yaml::Value) -> Result<Self, Error>;
        #[doc(hidden)]
        fn __serde_yaml__from_str(s: &str) -> Result<Self, Error>;
    }

    macro_rules! derive_yaml_read {
        ($Type:ty) => {
            impl $crate::YamlRead for $Type {
                fn __serde_ignored__from_value(value: serde_yaml::Value) -> Result<$Type, Error> {
                    serde_ignored::deserialize(
                        value,
                        |path| warn!("Unused config item (possible typo?): {}", path),
                    ).map_err(Into::into)
                }

                fn __serde_yaml__from_str(s: &str) -> Result<$Type, Error> {
                    serde_yaml::from_str(s)
                        .map_err(Into::into)
                }
            }
        };
    }

    derive_yaml_read!{serde_yaml::Value}

    // (this also exists solely for codegen reasons)
    fn value_from_str(r: &str) -> Result<serde_yaml::Value, Error>
    { serde_yaml::from_str(r).map_err(Into::into) }
}

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
