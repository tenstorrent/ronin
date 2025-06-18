#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

NAME=conv

CXX=/usr/lib/llvm-17/bin/clang++

JITTE=../../../../../jitte
TANTO=../../../../../tanto

SRC=../../../src
LIB=../../lib
BIN=../../bin

mkdir -p $BIN/$NAME

$CXX -std=c++20 -stdlib=libstdc++ -O3 -o $BIN/$NAME/test_tanto \
    -I $SRC/$NAME \
    -I $SRC/common \
    -I $TANTO/src \
    $SRC/$NAME/test/tanto/*.cpp \
    $LIB/$NAME/host_tanto.a \
    $LIB/$NAME/host_ref.a \
    $LIB/common/host_base.a \
    $LIB/common/host_util.a \
    $LIB/common/test_util.a \
    $TANTO/jitte/lib/host/core.a \
    $JITTE/lib/tt_metal/tt_metal.a \
    $JITTE/lib/tt_metal/tt_metal_impl.a \
    $JITTE/lib/tt_metal/tt_metal_detail.a \
    $JITTE/lib/tt_metal/jit_build.a \
    $JITTE/lib/tt_metal/common.a \
    $JITTE/lib/tt_metal/llrt.a \
    $JITTE/lib/tt_metal/emulator.a \
    $JITTE/lib/tt_metal/device.a \
    $JITTE/lib/tt_metal/yaml_cpp.a \
    $JITTE/lib/device/api.a \
    $JITTE/lib/device/dispatch.a \
    $JITTE/lib/device/ref.a \
    $JITTE/lib/device/riscv.a \
    $JITTE/lib/device/core.a \
    $JITTE/lib/device/arch.a \
    $JITTE/lib/device/schedule.a \
    $JITTE/lib/whisper/riscv.a \
    $JITTE/lib/whisper/linker.a \
    $JITTE/lib/whisper/interp.a


