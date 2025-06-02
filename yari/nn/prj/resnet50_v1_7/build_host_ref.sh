#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=resnet50_v1_7

CXX=/usr/lib/llvm-17/bin/clang++

OP=../../../op

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -I $SRC/$NAME \
    -I $SRC/common \
    -I $OP/src/common \
    -I $SRC/vendor \
    $SRC/$NAME/host/ref/*.cpp

mkdir -p $LIB/$NAME

ar rsc $LIB/$NAME/host_ref.a *.o

rm *.o


