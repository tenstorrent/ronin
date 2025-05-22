#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

METAL=$TT_METAL_HOME
VENDOR=../../vendor
BOOST=$VENDOR/boost

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libc++ -O3 \
    -D FMT_HEADER_ONLY \
    -D METAL_057 \
    -I $SRC/host \
    -I $METAL \
    -I $METAL/tt_metal/api \
    -I $METAL/tt_metal/hostdevcommon/api \
    -I $METAL/tt_metal/third_party/umd/device/api \
    -I $METAL/tt_stl \
    -I $VENDOR \
    -I $VENDOR/reflect \
    -I $BOOST/core/include \
    $SRC/host/core/*.cpp

mkdir -p $LIB/host

ar rsc $LIB/host/core.a *.o

rm *.o


