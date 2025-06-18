#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -I $SRC \
    -I $SRC/device \
    $SRC/device/ref/*.cpp

mkdir -p $LIB/device

ar rsc $LIB/device/ref.a *.o

rm *.o


