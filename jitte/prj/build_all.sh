#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Build tt_metal"
cd ./tt_metal
./build_all.sh
cd ..

echo "Build device"
cd ./device
./build_all.sh
cd ..

echo "Build whisper"
cd ./whisper
./build_all.sh
cd ..

echo "Build examples"
cd ./examples
./build_all.sh
cd ..


