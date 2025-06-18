#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=resnet50_v1_7

CXX=/usr/lib/llvm-17/bin/clang++

OP=../../../../op

SRC=../../../src
LIB=../../lib
BIN=../../bin

mkdir -p $BIN/$NAME

$CXX -std=c++20 -stdlib=libstdc++ -O3 -o $BIN/$NAME/test_ref \
    -I $SRC/$NAME \
    -I $SRC/common \
    -I $OP/src/common \
    $SRC/$NAME/test/ref/*.cpp \
    $LIB/$NAME/host_ref.a \
    $LIB/common/host_ref.a \
    $LIB/common/test_util.a \
    $LIB/vendor/arhat/runtime.a \
    $OP/lib/binary/host_ref.a \
    $OP/lib/conv/host_ref.a \
    $OP/lib/fc/host_ref.a \
    $OP/lib/group_conv/host_ref.a \
    $OP/lib/pool/host_ref.a \
    $OP/lib/reduce/host_ref.a \
    $OP/lib/common/host_base.a \
    $OP/lib/common/host_util.a


