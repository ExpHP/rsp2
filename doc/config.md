# `rsp2` configuration

## Config arguments

RSP2 accepts multiple config arguments in a single command line. When multiple configurations are specified, they are merged together, with values from later arguments overriding earlier ones.

```yaml
rsp2 -o outdir structure.vasp -c site.yaml -c structure-settings.yaml -c potential-kcz.yaml
```

### Config files and config literals

In addition to config files, it also allows YAML literals to be provided directly on the command line. This is implemented because it is often useful while debugging (as many rapid changes to configuration may be necessary).

```yaml
# (note: this also uses the dotted path syntax, described later)
rsp2 -o outdir structure.vasp -c settings.yaml -c 'lammps-update-style.fast.interval: 5'
```

The syntactic difference between a config literal and a config file path is that a config literal contains a colon (`:`).  Basically, any string with a `:` will be handed directly to a YAML parser (with one special consideration: a *leading* colon is stripped to simulate a dotted path of length 0, for writing things like `:{a: 1, b: 2}`).

### Merging

The rules of config merging are very simple.  At any given path inside a yaml file:

* If both config files contain a mapping at that path, they are merged recursively.
  * There is one exception to this (see the section on `~~REPLACE~~` singletons)
* If both config files contain a value at that path, but at least one is not a mapping, then only the newer value is used.

#### Examples:

```yaml
# File 1
potential:
  airebo:
    lj-sigma: 3

# File 2
potential:
  airebo:
    lj-enabled: true

# Merged result
potential:
  airebo:
    lj-sigma: 3
    lj-enabled: true
```

```yaml
# File 1
lammps-axis-mask: [true, true, true]

# File 2
lammps-axis-mask: [true, true, false]

# Merged result
# (newer value takes precedence because it is a sequence,
#  and only mappings are merged)
lammps-axis-mask: [true, true, false]
```

```yaml
# File 1
potential: rebo

# File 2
potential:
  airebo: {}

# Merged result
# (newer value takes precedence because the first one
#  was a string and not a mapping.  Note it is not generally
#  recommended to rely on this for fields like "potential";
#  see the section on ~~REPLACE~~ for a more reliable strategy)
potential:
  airebo: {}
```

### Seeing the results

When rsp2 runs, it writes these files to the output directory:

* `input-config-sources.yaml`: A file with the contents of each config file or config literal, in the order they were supplied.
* `settings.yaml`: The final effective YAML file after merging everything.

## Extensions to YAML syntax

### dotted keys

If a key in a yaml mapping contains a `.`, it is expanded to a nested singleton mapping. This is primarily useful for config literals on the command line. (It is also useful for the "replacement" extension, described next)

```yaml
# Raw input
a.b.c.d: 1

# Interpreted as
a:
  b:
    c:
      d: 1
```

The produced mappings will be merged with other mappings, as long as no path to a non-mapping value is defined twice:

```yaml
# Raw input
a.b1: 1
a:
  b2: 2

# Interpreted as
a:
  b1: 1
  b2: 2
```

```yaml
# Forbidden (multiple definitions for a.b1)
a.b1: 1
a:
  b1: 1
```

No means of escaping a dot is provided because none is necessary; `rsp2` does not use dots in its config keys!

### `~~REPLACE~~` singletons

Many of rsp2's config sections take the form of a rust enum serialized as a singleton mapping, whose key determines the variant. Unfortunately, this does not play very well with our dumb merging scheme:

```yaml
# File 1
potential:
  airebo: {}

# File 2
potential:
  kc-z: {}

# Result after merging. Oops!
potential:
  airebo: {}
  kc-z: {}
```

To work around this, you can use a singleton mapping with the key `~~REPLACE~~` to force a mapping to replace a value entirely, rather than being merged:

```yaml
# File 1
potential:
  airebo: {}

# File 2
potential:
  ~~REPLACE~~:
    kc-z: {}

# Result after merging
potential:
  kc-z: {}
```

Dotted keys can also be used to make this more compact:

```yaml
# File 2, more compactly
potential.~~REPLACE~~:
  kc-z: {}
```

`~~REPLACE~~` singletons are permitted even in the first config file (where they are effectively a no-op).
