#!/bin/sh

# openmpi version 3 is required by rsp2 (and the mpi crate in general)
# but the apt repos only have version 1

# we can't use 'cache: ccache' with 'language: python', so we manually cache it.
# This saves ~20 minutes of build time, and the cached data is only a few MB.
CACHE=$HOME/cache/openmpi-${OPENMPI_VERSION}

[ -e $CACHE ] || {
    cd $HOME/builds || exit 1
    wget -q https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_SHORT}/openmpi-${OPENMPI_VERSION}.tar.bz2 || exit 1
    tar -xf openmpi-${OPENMPI_VERSION}.tar.bz2 || exit 1
    cd openmpi-$OPENMPI_VERSION || exit 1
    ./configure --prefix=$CACHE  || exit 1
    make -j2 install || exit 1
}

stow -t $HOME/deps -d $(dirname $CACHE) $(basename $CACHE)
