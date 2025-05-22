#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=interp

CXX=/usr/lib/llvm-17/bin/clang++

METAL=$TT_METAL_HOME
TANTO=../../../../tanto

SRC=../../src
LIB=../../lib
BIN=../../bin

mkdir -p $BIN/$NAME

$CXX -std=c++20 -stdlib=libc++ -O3 -o $BIN/$NAME/test_tanto \
    -I $SRC/$NAME \
    -I $SRC/common \
    -I $TANTO/src \
    $SRC/$NAME/test/tanto/*.cpp \
    $LIB/$NAME/host_tanto.a \
    $LIB/$NAME/host_ref.a \
    $LIB/common/host_util.a \
    $LIB/common/test_util.a \
    $TANTO/lib/host/core.a \
    -L $METAL/build/lib \
    -ltt_metal \
    -lc++


