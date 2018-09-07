#!/bin/sh

# openmpi version 3 is required by rsp2 (and the mpi crate in general)
# but the apt repos only have version 1

# It takes forever to build, even with ccache enabled.
# So we explicitly cache it.
INSTALL=$HOME/cache/openmpi-${OPENMPI_VERSION}

[ -e $INSTALL ] || {
    cd $HOME/builds
    wget -q https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_SHORT}/openmpi-${OPENMPI_VERSION}.tar.bz2
    tar -xf openmpi-${OPENMPI_VERSION}.tar.bz2
    cd openmpi-$OPENMPI_VERSION
    ./configure --prefix=$INSTALL
    make -j2 install
}

stow -t $HOME/deps -d $(dirname $INSTALL) $(basename $INSTALL)
