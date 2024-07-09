#!/bin/bash
mkdir -p build
cd build || exit
cmake ..
make -j4
cd bin || exit
./CCF_QDALS