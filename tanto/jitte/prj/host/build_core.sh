#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

JITTE=../../../../jitte

SRC=../../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -Wno-deprecated-this-capture \
    -D TENSIX_FIRMWARE \
    -D FMT_HEADER_ONLY \
    -D YAML_CPP_STATIC_DEFINE \
    -I $SRC/host \
    -I $JITTE/src \
    -I $JITTE/src/tt_metal \
    -I $JITTE/src/tt_metal/hw/inc \
    -I $JITTE/src/tt_metal/emulator/hw \
    -I $JITTE/src/tt_metal/third_party/umd \
    -I $JITTE/src/tt_metal/third_party/fmt \
    -I $JITTE/src/yaml-cpp/include \
    -I $JITTE/src/system \
    $SRC/host/core/*.cpp

mkdir -p $LIB/host

ar rsc $LIB/host/core.a *.o

rm *.o


