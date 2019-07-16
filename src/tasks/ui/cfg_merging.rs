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

use crate::FailResult;

use rsp2_tasks_config::YamlRead;

use serde_yaml::{Value, Mapping};
use path_abs::{PathFile, FileRead};
use std::path::Path;

pub const CONFIG_HELP_STR: &'static str = "\
    config yaml, provided as either a filepath, or as an embedded literal \
    (via syntax described below). \
    When provided multiple times, the configs are merged according to some fairly \
    dumb strategy, with preference to the values supplied in later arguments. \
    \n\n\
    rsp2 supports some small extensions on the YAML syntax.  See `doc/config.md`.
    \n\n\
    Literals are written as '--config [KEY]:VALID_YAML', \
    where KEY is an optional string key (which may be dotted, as described in config.md), \
    and the ':' is a literal colon. When provided, KEY constructs a singleton \
    mapping (so `--config a.b.c:[2]` is equivalent to `--config ':{a: {b: {c: [2]}}}'`.\
    \n\n\
    Note that detection of filepaths versus literals is based solely \
    on the presence of a colon, and no means of escaping one in a path \
    are currently provided.\
";

pub const CONFIG_OVERRIDE_HELP_STR: &'static str = "\
    additional config yamls used only for this step.  If provided, these will be merged on top of \
    the `settings.yaml` saved inside the trial directory from the initial run to provide a new \
    effective configuration for this step.\
    \n\n\
    See `rsp2 --help` for more information about the config argument syntax.\
";

/// A list of config yamls that can be merged into a single effective config.
///
/// Can be serialized to a file that shows all of the configs in detail.
#[derive(Serialize)]
#[derive(Debug, Clone)]
pub struct ConfigSources(Vec<Config>);

// entry in config-sources.yaml
#[derive(Serialize)]
#[derive(Debug, Clone)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct Config {
    source: ConfigSource,
    yaml: ConflictFree,
}

#[derive(Serialize)]
#[derive(Debug, Clone)]
#[serde(rename_all = "kebab-case")]
enum ConfigSource {
    File(PathFile),
    Argument,
}

impl Config {
    /// May do path resolution and file IO
    pub(crate) fn resolve_from_arg(s: &str) -> FailResult<Config>
    { resolve_from_arg::resolve_from_arg(s) }

    fn read_file(path: impl AsRef<Path>) -> FailResult<Config> {
        Self::_read_file(path.as_ref())
    }

    fn _read_file(path: &Path) -> FailResult<Config> {
        let path = PathFile::new(path)?;
        let yaml = YamlRead::from_reader(FileRead::open(&path)?)?;
        let yaml = expand_dot_keys(yaml)?;
        let yaml = validate_replacements_from_one_config(yaml)?;

        let source = ConfigSource::File(path);
        Ok(Config { yaml, source })
    }
}

mod resolve_from_arg {
    use super::*;

    // May do path resolution and file IO
    pub(crate) fn resolve_from_arg(s: &str) -> FailResult<Config> {
        // NOTE: unapologetically, no mechanism is provided for escaping a path containing ':'.
        if s.contains(":") { lit_from_arg(s) }
        else { Config::read_file(s) }
    }

    fn lit_from_arg(s: &str) -> FailResult<Config> {
        let mut it = s.splitn(2, ":");

        let key = match it.next() {
            None => panic!("BUG! splitn always produces at least one string"),
            Some("") => None,
            Some(key) => Some(key),
        };
        let value = {
            let s = it.next().expect("BUG! lit_from_arg called on string without ':'");
            YamlRead::from_reader(s.as_bytes())?
        };
        let yaml = match key {
            Some(key) => make_singleton(key, value),
            None => value,
        };
        let yaml = expand_dot_keys(yaml)?;
        let yaml = validate_replacements_from_one_config(yaml)?;

        let source = ConfigSource::Argument;
        Ok(Config { yaml, source })
    }

    #[cfg(test)]
    macro_rules! yaml {
        ({ $($key:tt : $value:tt),*$(,)* }) => {{
            let items = vec![$((yaml!($key), yaml!($value))),*];
            let mapping = items.into_iter().collect();
            Value::Mapping(mapping)
        }};
        ([ $($value:tt),*$(,)* ]) => {{
            let items = vec![$(yaml!($value)),*];
            Value::Sequence(items)
        }};
        ($e:tt) => { $e.into() }
    }

    #[test]
    fn test_literal_args() {
        macro_rules! expect {
            ($expected:expr, $s:expr) => {{
                let cfg = Config::resolve_from_arg($s).unwrap();
                let ConflictFree(DotFree(yaml)) = cfg.yaml;
                assert_eq!{$expected, yaml}
            }};
        }
        macro_rules! expect_replace_error {
            ($s:expr) => {
                let e = Config::resolve_from_arg($s).unwrap_err();
                match e.downcast() {
                    Err(_) => panic!($s),
                    Ok(InternalReplaceError) => {},
                }
            };
        }

        let expected = yaml!{{
            "hello": {
                "how-are": {
                    "you": [42],
                },
            },
        }};
        expect!(expected, ":{hello: {how-are: {you: [42]}}}");
        expect!(expected, ": {hello: {how-are: {you: [42]}}}");
        expect!(expected, "hello:{how-are: {you: [42]}}");
        expect!(expected, "hello: {how-are: {you: [42]}}");
        expect!(expected, "hello.how-are: {you: [42]}");
        expect!(expected, "hello.how-are.you: [42]");
        expect!(expected, "hello: {how-are.you: [42]}");
        expect!(expected, ":{hello.how-are.you: [42]}");

        let expected = yaml!{{
            "a": { "b1": 1, "b2": 2 },
        }};
        expect!(expected, ": {a: {b1: 1, b2: 2}}");
        expect!(expected, ": {a.b1: 1, a.b2: 2}");
        expect!(expected, "a: {b1: 1, b2: 2}");
        expect!(expected, ": {a: {b1: 1}, a.b2: 2}");

        expect!(
            yaml!{{ "a": { "b1": {"~~REPLACE~~": 1}, "b2": 2 } }},
            ": {a: {b1: {~~REPLACE~~: 1}}, a.b2: 2}"
        );
        expect!(
            yaml!{{"~~REPLACE~~": {"a": { "b1": 1, "b2": 2 } }}},
            "~~REPLACE~~: {a: {b1: 1}, a.b2: 2}"
        );

        expect_replace_error!("a: {b1: 1, ~~REPLACE~~: 2}");
        expect_replace_error!(": {a.b1: 1, a.~~REPLACE~~: 2}");
    }
}

impl ConfigSources {
    /// Construct from values given to --config.
    ///
    /// # Notice
    /// Relative paths will be resolved immediately, and possibly
    /// even opened, read, and parsed as yaml.
    pub fn resolve_from_args<As>(args: As) -> FailResult<Self>
    where
        As: IntoIterator,
        As::Item: AsRef<str>,
    {
        let mut out = vec![];
        for arg in args {
            out.push(Config::resolve_from_arg(arg.as_ref())?);
        }
        Ok(ConfigSources(out))
    }

    pub fn into_effective_yaml(self) -> Value {
        let FullyResolved(ConflictFree(DotFree(value))) = self._into_effective_yaml();
        value
    }

    fn _into_effective_yaml(self) -> FullyResolved {
        let empty = FullyResolved(ConflictFree(DotFree(Value::Mapping(Default::default()))));
        self.0.into_iter()
            .fold(empty, |a, b| merge_nodot_and_replace(a, b.yaml))
    }

    // prepend a file source, so that all of the sources in this struct are considered to be
    // patches to it.
    pub fn prepend_file(&mut self, path: impl AsRef<Path>) -> FailResult<()> {
        self.0.insert(0, Config::read_file(path)?);
        Ok(())
    }

    pub fn deserialize<T: YamlRead>(self) -> FailResult<T> {
        // (NOTE: This is a Rube Goldberg machine of yaml conversions, all to have nice error
        //        messages. It goes to a Value and then to a string (as that's the easiest way to
        //        get a Read, which is required to have value paths appear in error messages),
        //        from which it will be parsed back into a Value in rsp2-tasks-config, etc...)
        let value = self.into_effective_yaml();
        let s = serde_yaml::to_string(&value)?;
        YamlRead::from_reader(s.as_bytes())
    }
}

const REPLACE_DIRECTIVE_KEY: &'static str = "~~REPLACE~~";

//----------------
// Types to document post- and pre-conditions

/// Indicates that:
/// * all keys are strings, and no keys contain dots
#[derive(Serialize)]
#[derive(Debug, Clone)]
struct DotFree(Value);

/// Indicates that:
/// * all keys are strings, and no keys contain dots
/// * any mapping with a REPLACE directive is a singleton.
#[derive(Serialize)]
#[derive(Debug, Clone)]
struct ConflictFree(DotFree);

/// Indicates that:
/// * all keys are strings, and no keys contain dots
/// * there are no REPLACE directives
#[derive(Serialize)]
#[derive(Debug, Clone)]
struct FullyResolved(ConflictFree);
//----------------

#[derive(Debug, Fail)]
#[fail(display = "\
    Invalid use of REPLACE directive; a REPLACE directive cannot be used \
    to replace other values in the same config file, as the keys may be \
    visited in a nondeterministic order.\
")]
pub struct InternalReplaceError;

/// A simplistic config-merging function which operates directly on the yaml representation,
/// independent of what is being deserialized. This sort of approach is inherently flawed,
/// but easiest to implement.
///
/// Given two mappings, it takes the union of their keys and recursively merges their intersection.
/// Given any other two values, it prefers 'b'.
///
/// It will have trouble overriding the values of struct-like enum variants due to their
/// representation as a mapping with one key.
fn dumb_config_merge(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Mapping(mut a), Value::Mapping(b)) => {
            for (key, b_value) in b {
                let value = match a.remove(&key) {
                    None => b_value,
                    Some(a_value) => dumb_config_merge(a_value, b_value),
                };
                a.insert(key, value);
            }
            Value::Mapping(a)
        },
        (_, b) => b,
    }
}

fn merge_nodot(a: DotFree, b: DotFree) -> DotFree
{ DotFree(dumb_config_merge(a.0, b.0)) }

fn merge_nodot_and_replace(a: FullyResolved, b: ConflictFree) -> FullyResolved {
    let out = merge_nodot((a.0).0, b.0);
    resolve_replacements_from_two_configs(out)
}

fn expect_string_key(value: Value) -> FailResult<String> {
    match value {
        Value::String(s) => Ok(s),
        _ => bail!{"yaml contains non-string key"},
    }
}

fn expand_dot_keys(value: Value) -> FailResult<DotFree> {
    match value {
        Value::Mapping(mapping) => {
            mapping.into_iter()
                .try_fold(
                    Value::Mapping(Default::default()),
                    |acc, (key, child)| {
                        let DotFree(child) = expand_dot_keys(child)?;

                        let key = expect_string_key(key)?;
                        let path: Vec<_> = key.split(".").collect();
                        let new_part = make_nested_mapping(&path, child);
                        Ok(dumb_config_merge(acc, new_part))
                    },
                )
        },
        Value::Sequence(values) => Ok(Value::Sequence({
            values.into_iter()
                .map(|v| expand_dot_keys(v).map(|DotFree(v)| v))
                .collect::<Result<_, _>>()?
        })),
        value => Ok(value),
    }.map(DotFree)
}

// resolve REPLACE directives, assuming that they all came from singletons in the second
// config file of a dumb merge.
fn resolve_replacements_from_two_configs(value: DotFree) -> FullyResolved
{ _resolve_replacements(value, false, false).map(FullyResolved).expect("(BUG!)") }

// validate but don't resolve REPLACE directives, assuming that they came from dot expansion on a
// single config
fn validate_replacements_from_one_config(value: DotFree) -> FailResult<ConflictFree>
{ _resolve_replacements(value, true, true) }

fn _resolve_replacements(
    value: DotFree,
    is_single_config: bool,
    dry_run: bool,
) -> FailResult<ConflictFree> {
    fold_mappings_depth_first(
        value.0,
        |mapping| Ok({
            let has_multiple_keys = mapping.len() > 1;
            let has_replace_key = mapping.contains_key(&Value::String(REPLACE_DIRECTIVE_KEY.into()));
            if has_multiple_keys && has_replace_key && is_single_config {
                Err(InternalReplaceError)?;
            }

            if has_replace_key && !dry_run {
                // If we have made it this far, all we need to do is substitute the mapping
                // with the value under the REPLACE key (regardless of whatever else the
                // mapping contains)
                mapping.into_iter()
                    .find(|(key, _)| key == REPLACE_DIRECTIVE_KEY)
                    .unwrap().1
            } else {
                Value::Mapping(mapping)
            }
        }),
    ).map(DotFree).map(ConflictFree)
}

fn fold_mappings_depth_first(
    value: Value,
    mut f: impl FnMut(Mapping) -> FailResult<Value>,
) -> FailResult<Value>
{ _fold_mappings_depth_first(value, &mut f) }

fn _fold_mappings_depth_first(
    value: Value,
    f: &mut impl FnMut(Mapping) -> FailResult<Value>,
) -> FailResult<Value>
{Ok({
    match value {
        Value::Mapping(mapping) => {
            let mapping = {
                mapping.into_iter()
                    .map(|(key, child)| Ok((key, _fold_mappings_depth_first(child, f)?)))
                    .collect::<FailResult<_>>()?
            };
            f(mapping)?
        },
        Value::Sequence(values) => Value::Sequence({
            values.into_iter()
                .map(|value| _fold_mappings_depth_first(value, f))
                .collect::<FailResult<_>>()?
        }),
        value => value,
    }
})}

// FIXME: These functions for summary.yaml probably don't belong here,
//        but `merge_summaries` is here to use a private function.
/// Merges summary.yaml files for output.
pub fn merge_summaries(a: Value, b: Value) -> Value {
    // Reuse the dumb algorithm because is actually perfect for this use case.
    // Summaries should continue to use the dumb algorithm even if config files
    // get a redesigned algorithm at some point.
    dumb_config_merge(a, b)
}

/// Make an empty summary.
pub fn no_summary() -> Value { Value::Mapping(Default::default()) }

/// Constructs a Yaml like `{a: {b: {c: value}}}`
pub fn make_nested_mapping(path: &[impl AsRef<str>], value: Value) -> Value {
    path.iter().rev().fold(value, |value, key| make_singleton(key, value))
}

/// Constructs a Yaml like `{a: value}`
fn make_singleton(key: impl AsRef<str>, value: Value) -> Value {
    let mut mapping = Mapping::new();
    mapping.insert(Value::String(key.as_ref().to_string()), value);
    Value::Mapping(mapping)
}
