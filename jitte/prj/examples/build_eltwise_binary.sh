#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=eltwise_binary

CXX=/usr/lib/llvm-17/bin/clang++

SRC=../../src
LIB=../../lib
BIN=../../bin

mkdir -p $BIN/examples

$CXX -o $BIN/examples/$NAME -std=c++20 -stdlib=libstdc++ -O3 \
    -Wno-deprecated-this-capture \
    -I $SRC \
    -I $SRC/tt_metal \
    -I $SRC/tt_metal/hw/inc \
    -I $SRC/tt_metal/emulator/hw \
    -I $SRC/tt_metal/third_party/umd \
    -I $SRC/tt_metal/third_party/fmt \
    -I $SRC/yaml-cpp/include \
    -DTENSIX_FIRMWARE \
    -DFMT_HEADER_ONLY \
    -DYAML_CPP_STATIC_DEFINE \
    $SRC/tt_metal/programming_examples/$NAME/*.cpp \
    $LIB/tt_metal/tt_metal.a \
    $LIB/tt_metal/tt_metal_impl.a \
    $LIB/tt_metal/tt_metal_detail.a \
    $LIB/tt_metal/jit_build.a \
    $LIB/tt_metal/common.a \
    $LIB/tt_metal/llrt.a \
    $LIB/tt_metal/emulator.a \
    $LIB/tt_metal/device.a \
    $LIB/tt_metal/yaml_cpp.a \
    $LIB/device/api.a \
    $LIB/device/dispatch.a \
    $LIB/device/ref.a \
    $LIB/device/riscv.a \
    $LIB/device/core.a \
    $LIB/device/arch.a \
    $LIB/device/schedule.a \
    $LIB/whisper/riscv.a \
    $LIB/whisper/linker.a \
    $LIB/whisper/interp.a

 
