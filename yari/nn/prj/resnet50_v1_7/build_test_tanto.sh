#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=resnet50_v1_7

CXX=/usr/lib/llvm-17/bin/clang++

METAL=$TT_METAL_HOME
TANTO=../../../../tanto
OP=../../../op

SRC=../../src
LIB=../../lib
BIN=../../bin

mkdir -p $BIN/$NAME

$CXX -std=c++20 -stdlib=libstdc++ -O3 -o $BIN/$NAME/test_tanto \
    -I $SRC/$NAME \
    -I $SRC/common \
    -I $OP/src/common \
    -I $TANTO/src \
    $SRC/$NAME/test/tanto/*.cpp \
    $LIB/$NAME/host_tanto.a \
    $LIB/$NAME/host_ref.a \
    $LIB/common/host_tanto.a \
    $LIB/common/host_ref.a \
    $LIB/common/test_util.a \
    $LIB/vendor/arhat/runtime.a \
    $OP/lib/binary/host_tanto.a \
    $OP/lib/binary/host_ref.a \
    $OP/lib/conv/host_tanto.a \
    $OP/lib/conv/host_ref.a \
    $OP/lib/fc/host_tanto.a \
    $OP/lib/fc/host_ref.a \
    $OP/lib/group_conv/host_tanto.a \
    $OP/lib/group_conv/host_ref.a \
    $OP/lib/pool/host_tanto.a \
    $OP/lib/pool/host_ref.a \
    $OP/lib/reduce/host_tanto.a \
    $OP/lib/reduce/host_ref.a \
    $OP/lib/common/host_base.a \
    $OP/lib/common/host_util.a \
    $TANTO/lib/host/core.a \
    -L $METAL/build/lib \
    -ltt_metal


