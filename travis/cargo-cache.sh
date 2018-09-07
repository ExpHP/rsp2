#!/bin/sh

# Can't use 'cache: cargo' with 'language: python'

CACHE=$HOME/cache/rsp2-cargo-stuffs

if [ x"$1" == xstore ]; then
    mkdir -p $CACHE

    # Keep build artefacts
    rm -rf $CACHE/target
    cp -a $TRAVIS_BUILD_DIR/target $CACHE

    # Maximize reuse of build artefacts by using previously built versions
    # of crates whenever possible.
    cp $TRAVIS_BUILD_DIR/Cargo.lock $CACHE

    # Don't cache ~/.cargo as Travis docs recommend against caching downloads.
    #
    # Don't worry; when .cargo is lost, the packages are re-downloaded, but not
    # rebuilt so long as the stuff in target/ is good.
elif [ x"$1" == xload ]; then
    cp $CACHE/Cargo.lock $TRAVIS_BUILD_DIR

    rm -rf $TRAVIS_BUILD_DIR/target
    cp -a $CACHE/target $TRAVIS_BUILD_DIR

    true
else
    echo >&2 "Bad call to cargo-cache.sh"
    exit 1
fi
