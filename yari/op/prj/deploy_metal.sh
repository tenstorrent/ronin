#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

mkdir -p $TT_METAL_HOME/op

mkdir -p $TT_METAL_HOME/op/binary
mkdir -p $TT_METAL_HOME/op/conv
mkdir -p $TT_METAL_HOME/op/fc
mkdir -p $TT_METAL_HOME/op/group_conv
mkdir -p $TT_METAL_HOME/op/interp
mkdir -p $TT_METAL_HOME/op/move
mkdir -p $TT_METAL_HOME/op/pool
mkdir -p $TT_METAL_HOME/op/reduce

cp -R -v ../src/binary/device $TT_METAL_HOME/op/binary
cp -R -v ../src/conv/device $TT_METAL_HOME/op/conv
cp -R -v ../src/fc/device $TT_METAL_HOME/op/fc
cp -R -v ../src/group_conv/device $TT_METAL_HOME/op/group_conv
cp -R -v ../src/interp/device $TT_METAL_HOME/op/interp
cp -R -v ../src/move/device $TT_METAL_HOME/op/move
cp -R -v ../src/pool/device $TT_METAL_HOME/op/pool
cp -R -v ../src/reduce/device $TT_METAL_HOME/op/reduce


