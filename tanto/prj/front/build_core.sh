#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

SRC=../../src
LIB=../../lib

LLVM_INC=/usr/lib/llvm-16/include

g++ -c -std=c++17 -O3 \
    -I $SRC/front \
    -I $LLVM_INC \
    $SRC/front/core/*.cpp

mkdir -p $LIB/front

ar rsc $LIB/front/core.a *.o

rm *.o


