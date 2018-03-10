use ::Result;

use ::rsp2_tasks_config::YamlRead;

use ::serde_yaml::{Value, Mapping};
use ::path_abs::PathFile;

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
    yaml: Value,
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
    pub(crate) fn resolve_from_arg(s: &str) -> Result<Config>
    { resolve_from_arg::resolve_from_arg(s) }
}

mod resolve_from_arg {
    use super::*;

    pub(crate) fn resolve_from_arg(s: &str) -> Result<Config> {
        // NOTE: unapologetically, no mechanism is provided for escaping a path containing ':'.
        if s.contains(":") { lit_from_arg(s) }
        else { read_file_from_arg(s) }
    }

    fn lit_from_arg(s: &str) -> Result<Config> {
        let mut it = s.splitn(2, ":");

        let path = match it.next() {
            None => panic!("BUG! splitn always produces at least one string"),
            Some("") => vec![],
            Some(path) => path.split(".").collect(),
        };
        let value = {
            let s = it.next().expect("BUG! lit_from_arg called on string without ':'");
            YamlRead::from_reader(s.as_bytes())?
        };
        let yaml = make_nested_mapping(&path, value);
        let source = ConfigSource::Argument;

        Ok(Config { yaml, source })
    }

    // May do path resolution and file IO
    fn read_file_from_arg(path: &str) -> Result<Config> {
        let path = PathFile::new(path)?;
        let yaml = YamlRead::from_reader(path.read()?)?;
        let source = ConfigSource::File(path);
        Ok(Config { yaml, source })
    }

    #[cfg(test)]
    macro_rules! m { ($($arg:tt)*) => { Value::Mapping(vec![$($arg)*].into_iter().collect()) }; }
    #[cfg(test)]
    macro_rules! s { ($($arg:tt)*) => { Value::Sequence(vec![$($arg)*]) }; }

    #[test]
    fn test_literal_args() {
        let expected = m!{ ("hello".into(), m!{ ("how-are-you".into(), s![42.into()]) }) };
        assert_eq!(expected, Config::resolve_from_arg(":{hello: {how-are-you: [42]}}").unwrap().yaml);
        assert_eq!(expected, Config::resolve_from_arg(": {hello: {how-are-you: [42]}}").unwrap().yaml);
        assert_eq!(expected, Config::resolve_from_arg("hello:{how-are-you: [42]}").unwrap().yaml);
        assert_eq!(expected, Config::resolve_from_arg("hello: {how-are-you: [42]}").unwrap().yaml);
        assert_eq!(expected, Config::resolve_from_arg("hello.how-are-you: [42]").unwrap().yaml);
    }
}

impl ConfigSources {
    /// Construct from values given to --config.
    ///
    /// # Notice
    /// Relative paths will be resolved immediately, and possibly
    /// even opened, read, and parsed as yaml.
    pub fn resolve_from_args<As>(args: As) -> Result<Self>
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
        let empty = Value::Mapping(Default::default());
        self.0.into_iter()
            .fold(empty, |a, b| dumb_config_merge(a, b.yaml))
    }
}

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
pub fn make_nested_mapping(path: &[&str], value: Value) -> Value {
    path.iter().rev().fold(value, |value, &key| {
        let mut mapping = Mapping::new();
        mapping.insert(Value::String(key.into()), value);
        Value::Mapping(mapping)
    })
}
