#!/bin/sh

# Can't use 'cache: cargo' with 'language: python'

CACHE=$HOME/cache/rsp2-cargo-stuffs

if [ "$1" = store ]; then
    mkdir -p $CACHE

    # Keep build artefacts
    rm -rf $CACHE/target
    cp -a $TRAVIS_BUILD_DIR/target $CACHE
    (
        # archive because otherwise travis' cacher will try to compute
        # a bajillion MD5 sums and timeout.
        cd $CACHE
        tar -cf target.tar target
        rm -rf target
    )

    # Don't cache ~/.cargo as Travis docs recommend against caching downloads.
    #
    # Don't worry; when .cargo is lost, the packages are re-downloaded, but not
    # rebuilt so long as the stuff in target/ is good.
elif [ "$1" = load ]; then
    rm -rf $TRAVIS_BUILD_DIR/target
    cp $CACHE/target.tar $TRAVIS_BUILD_DIR
    (
        cd $TRAVIS_BUILD_DIR
        tar -xf target.tar
        rm target.tar
    )

    true
else
    echo >&2 "Bad call to cargo-cache.sh"
    exit 1
fi
