#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Build vendor/arhat"
cd ./vendor/arhat
./build_runtime.sh
cd ../..

echo "Build common"
cd ./common
./build_all.sh
cd ..

echo "Build mobilenetv2_050"
cd ./mobilenetv2_050
./build_all.sh
cd ..

echo "Build resnet18"
cd ./resnet18
./build_all.sh
cd ..

echo "Build resnet50_v1_7"
cd ./resnet50_v1_7
./build_all.sh
cd ..

echo "Done"


