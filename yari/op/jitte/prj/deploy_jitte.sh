#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

export JITTE_HOME=../../../../jitte/home

mkdir -p $JITTE_HOME/op

mkdir -p $JITTE_HOME/op/binary
mkdir -p $JITTE_HOME/op/conv
mkdir -p $JITTE_HOME/op/deform_conv
mkdir -p $JITTE_HOME/op/fc
mkdir -p $JITTE_HOME/op/group_conv
mkdir -p $JITTE_HOME/op/interp
mkdir -p $JITTE_HOME/op/move
mkdir -p $JITTE_HOME/op/pool
mkdir -p $JITTE_HOME/op/reduce

cp -R -v ../../src/binary/device $JITTE_HOME/op/binary
cp -R -v ../../src/conv/device $JITTE_HOME/op/conv
cp -R -v ../../src/deform_conv/device $JITTE_HOME/op/deform_conv
cp -R -v ../../src/fc/device $JITTE_HOME/op/fc
cp -R -v ../../src/group_conv/device $JITTE_HOME/op/group_conv
cp -R -v ../../src/interp/device $JITTE_HOME/op/interp
cp -R -v ../../src/move/device $JITTE_HOME/op/move
cp -R -v ../../src/pool/device $JITTE_HOME/op/pool
cp -R -v ../../src/reduce/device $JITTE_HOME/op/reduce


