#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

SRC=../../src
LIB=../../lib
BIN=../../bin

LLVM_INC=/usr/lib/llvm-16/include
LLVM_LIB=/usr/lib/llvm-16/lib

mkdir -p $BIN/front

g++ -std=c++17 -o $BIN/front/tanto \
    -I $SRC/front \
    -I $LLVM_INC \
    $SRC/front/cmd/*.cpp \
    $LIB/front/core.a \
    -L $LLVM_LIB \
    -l clang-cpp \
    -l LLVM

 
