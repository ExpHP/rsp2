#!/bin/sh

set -eux

CACHE=$HOME/cache/dftbplus-${DFTBPLUS_REV}

[[ -e $CACHE ]] || {
    PREFIX=$CACHE
    DATAROOTDIR=$PREFIX/usr/share
    DATADIR=$DATAROOTDIR
    INCLUDEDIR=$PREFIX/include
    LIBDIR=$PREFIX/lib

    cd ~/build
    git clone https://github.com/dftbplus/dftbplus
    cd dftbplus

    # static lib dependencies. Determined by seeing what 'make test_api' uses
    # for its own C-language tests.
    OTHERLIBS_C="-lgfortran -lm -lgomp"
    LIB_LAPACK="-lopenblas"
    API_VERSION=$(cat api/mm/API_VERSION)
    GIT_VERSION=$(git describe --long | sed 's/\([^-]*-\)g/r\1/;s/-/./g')

    # Perform the build
    ./utils/get_opt_externals
    cp sys/make.x86_64-linux-gnu make.arch
    make BUILD_API=1 LIB_LAPACK="$LIB_LAPACK" OTHERLIBS_C="$OTHERLIBS_C" -j "$(nproc)" api
    make INSTALLDIR="$PREFIX" install_api

    # install_api is a bit messy and incomplete.
    rm "$INCLUDEDIR"/*.mod # don't need all these fortran modules
    install -Dm644 LICENSE "$DATAROOTDIR/licenses/dftbplus/LICENSE"
    install -Dm644 api/mm/dftbplus.h "$PREFIX/dftbplus.h"
    install -D _build/external/xmlf90/libxmlf90.a "$LIBDIR/libxmlf90.a"

    mkdir -p "$LIBDIR/pkgconfig"
    cat >"$LIBDIR/pkgconfig/libdftb+.pc" <<HERE
lib_lapack=$LIB_LAPACK
lib_other_c=$OTHERLIBS_C
api_version=$API_VERSION
git_version=$GIT_VERSION

prefix=/usr/local
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: DFTB+
Description: C library for DFTB+, a fast and efficient versatile quantum mechanical simulation software package.
Version: \${git_version}
Requires.private:
Libs: -L\${libdir} -ldftb+ -lxmlf90
Libs.private: \${lib_lapack} \${lib_other_c}
Cflags: -I\${includedir}
HERE
}

stow -t $HOME/deps -d $(dirname $CACHE) $(basename $CACHE)
