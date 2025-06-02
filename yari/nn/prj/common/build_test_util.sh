#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=common

CXX=/usr/lib/llvm-17/bin/clang++

OP=../../../op

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -I $SRC/$NAME \
    -I $SRC/vendor \
    $SRC/$NAME/test/util/*.cpp

mkdir -p $LIB/$NAME

ar rsc $LIB/$NAME/test_util.a *.o

rm *.o


