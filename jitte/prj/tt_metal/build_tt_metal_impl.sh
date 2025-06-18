#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -Wno-deprecated-this-capture \
    -Wno-deprecated-volatile \
    -D TENSIX_FIRMWARE \
    -D FMT_HEADER_ONLY \
    -D YAML_CPP_STATIC_DEFINE \
    -I $SRC \
    -I $SRC/tt_metal \
    -I $SRC/tt_metal/impl \
    -I $SRC/tt_metal/hw/inc \
    -I $SRC/tt_metal/emulator/hw \
    -I $SRC/tt_metal/third_party/umd \
    -I $SRC/tt_metal/third_party/fmt \
    -I $SRC/yaml-cpp/include \
    -I $SRC/system \
    $SRC/tt_metal/impl/allocator/*.cpp \
    $SRC/tt_metal/impl/allocator/algorithms/*.cpp \
    $SRC/tt_metal/impl/buffers/*.cpp \
    $SRC/tt_metal/impl/device/*.cpp \
    $SRC/tt_metal/impl/dispatch/*.cpp \
    $SRC/tt_metal/impl/kernels/*.cpp \
    $SRC/tt_metal/impl/program/*.cpp \
    $SRC/tt_metal/impl/trace/*.cpp

mkdir -p $LIB/tt_metal

ar rsc $LIB/tt_metal/tt_metal_impl.a *.o

rm *.o


