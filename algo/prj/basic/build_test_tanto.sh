#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

METAL=$TT_METAL_HOME
TANTO=../../../tanto

SRC=../../src
LIB=../../lib
BIN=../../bin

mkdir -p $BIN/basic

$CXX -std=c++20 -stdlib=libc++ -O3 -o $BIN/basic/test_tanto \
    -I $SRC/basic \
    -I $TANTO/src \
    $SRC/basic/test/tanto/*.cpp \
    $LIB/basic/test_ref.a \
    $LIB/basic/test_util.a \
    $LIB/basic/host_tanto.a \
    $TANTO/lib/host/core.a \
    -L $METAL/build/lib \
    -ltt_metal \
    -lc++


