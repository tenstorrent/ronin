#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=deform_conv

CXX=/usr/lib/llvm-17/bin/clang++

TANTO=../../../../tanto

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -I $SRC/$NAME \
    -I $SRC/common \
    -I $TANTO/src \
    $SRC/$NAME/host/tanto/*.cpp

mkdir -p $LIB/$NAME

ar rsc $LIB/$NAME/host_tanto.a *.o

rm *.o


