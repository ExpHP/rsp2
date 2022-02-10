#![allow(non_snake_case)]

use failure::Error;

use std::io::Read;

/// Provides an alternative to `serde_yaml::from_reader` which can improve
/// compile time.
///
/// When using this trait, all of the expensive codegen happens in the crate which
/// provides the impl.  Thus, if a large deserializable type is moved into its own
/// dedicated crate, then downstream code in other crates which reads this type
/// will have much faster edit-recompile cycles.
///
/// It also uses `serde_ignored` to warn on unrecognized keys.
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

#[macro_export]
macro_rules! derive_yaml_read {
    ($Type:ty) => {
        const _: () = {
            use std::result::Result;
            use std::convert::Into;
            use $crate::reexports::serde_yaml;
            use $crate::reexports::serde_ignored;
            use failure::Error;
            use log::warn;

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
    };
}

derive_yaml_read!{serde_yaml::Value}

// (this also exists solely for codegen reasons)
fn value_from_str(r: &str) -> Result<serde_yaml::Value, Error>
{ serde_yaml::from_str(r).map_err(Into::into) }
