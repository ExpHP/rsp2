
__all__ = [
    'load',
    'dump',
    'load_all',
    'dump_all',
]

import functools

try:
    from ruamel import YAML as yaml
    yaml = yaml(typ='rt')
    load = yaml.load
    dump = yaml.dump
    load_all = yaml.load_all
    dump_all = yaml.dump_all
except ImportError:
    try:
        import yaml
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        load = functools.partial(yaml.load, Loader=Loader)
        dump = functools.partial(yaml.dump, Dumper=Dumper)
        load_all = functools.partial(yaml.load_all, Loader=Loader)
        dump_all = functools.partial(yaml.dump_all, Dumper=Dumper)
    except ImportError:
        raise ImportError('could not find ruamel or PyYAML')

