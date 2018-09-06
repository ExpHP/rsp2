#!/bin/sh

cd $HOME/builds || exit 1
wget -q https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_SHORT}/openmpi-${OPENMPI_VERSION}.tar.bz2 || exit 1
tar -xf openmpi-${OPENMPI_VERSION}.tar.bz2 || exit 1
cd openmpi-$OPENMPI_VERSION || exit 1

./configure --prefix=$HOME/deps || exit 1
make install >install.log 2>&1 || exit 1
