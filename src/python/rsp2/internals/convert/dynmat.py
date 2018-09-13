from . import _main_from_cli, _main_from_rust

def main_from_cli():
    from rsp2.io import dynmat

    _main_from_cli(
        from_path=dynmat.from_path,
        to_path=dynmat.to_path,
        equal=dynmat.equal,
    )

def main_from_rust():
    from rsp2.io import dynmat

    _main_from_rust(
        from_path=dynmat.from_path,
        to_path=dynmat.to_path,
        equal=dynmat.equal,
    )

if __name__ == '__main__':
    main_from_rust()
