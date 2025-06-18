#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

TANTO=../../../../tanto

SRC=../../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -I $SRC/basic \
    -I $TANTO/src \
    $SRC/basic/host/tanto/*.cpp

mkdir -p $LIB/basic

ar rsc $LIB/basic/host_tanto.a *.o

rm *.o


