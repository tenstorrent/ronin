#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Build common"
cd ./common
./build_all.sh
cd ..

echo "Build binary"
cd ./binary
./build_all.sh
cd ..

echo "Build conv"
cd ./conv
./build_all.sh
cd ..

echo "Build deform_conv"
cd ./deform_conv
./build_all.sh
cd ..

echo "Build fc"
cd ./fc
./build_all.sh
cd ..

echo "Build group_conv"
cd ./group_conv
./build_all.sh
cd ..

echo "Build interp"
cd ./interp
./build_all.sh
cd ..

echo "Build move"
cd ./move
./build_all.sh
cd ..

echo "Build pool"
cd ./pool
./build_all.sh
cd ..

echo "Build reduce"
cd ./reduce
./build_all.sh
cd ..

echo "Done"


