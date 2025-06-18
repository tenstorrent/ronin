#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXX=/usr/lib/llvm-17/bin/clang++

SRC=../../src
LIB=../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -D YAML_CPP_STATIC_DEFINE \
    -I $SRC/yaml-cpp/include \
    $SRC/yaml-cpp/src/*.cpp

mkdir -p $LIB/tt_metal

ar rsc $LIB/tt_metal/yaml_cpp.a *.o

rm *.o


