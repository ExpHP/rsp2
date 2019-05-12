#!/usr/bin/env bash

# Delete a bunch of executable binaries that are around 100Mb in size
# and only take moments to link, reducing the cache size.

delete() {
    for arg in "$@"; do
        if [ -f "$arg" ]; then
            echo "Deleting $arg"
            rm "$arg"
        fi
    done
}

for subdir in debug release; do
    target=./target/$subdir

    delete $target/deps/librsp2*
    delete $target/deps/rsp2*

    # Executable bit in deps/ is set for:
    # - the package's binaries (no extension)
    # - things in tests/ (no extension)
    # - proc macros (.so extension)
    #
    # Don't delete the proc-macros
    for name in $(ls $target/deps); do
        if [ -x "$target/deps/$name" -a "${name%.so}" = "$name" ]; then
            delete "$target/deps/$name"
        fi
    done

    # More copies in the parent directory
    for name in $(ls "$target"); do
        if [ -x "$target/$name" ]; then
            delete "$target/$name"
        fi
    done
done

echo
echo "Remaining stuff"
du --max-depth=3 --all "$target" || true
