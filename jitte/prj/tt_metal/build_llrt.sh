#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -Wno-deprecated-this-capture \
    -D TENSIX_FIRMWARE \
    -D FMT_HEADER_ONLY \
    -D YAML_CPP_STATIC_DEFINE \
    -D ARCH_WORMHOLE_B0 \
    -I $SRC \
    -I $SRC/tt_metal \
    -I $SRC/tt_metal/hw/inc \
    -I $SRC/tt_metal/emulator/hw \
    -I $SRC/tt_metal/third_party/umd \
    -I $SRC/tt_metal/third_party/fmt \
    -I $SRC/yaml-cpp/include \
    -I $SRC/system \
    $SRC/tt_metal/llrt/*.cpp \
    $SRC/tt_metal/llrt/blackhole/*.cpp \
    $SRC/tt_metal/llrt/grayskull/*.cpp \
    $SRC/tt_metal/llrt/wormhole/*.cpp


mkdir -p $LIB/tt_metal

ar rsc $LIB/tt_metal/llrt.a *.o

rm *.o


