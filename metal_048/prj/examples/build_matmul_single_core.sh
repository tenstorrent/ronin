#!/bin/bash

NAME=matmul_single_core

SRC=../../src
LIB=../../lib
BIN=../../bin

mkdir -p $BIN/examples

g++ -o $BIN/examples/$NAME -std=c++17 -O3 \
    -I $SRC \
    -I $SRC/tt_metal \
    -I $SRC/tt_metal/hw/inc \
    -I $SRC/tt_metal/hw/inc/wormhole \
    -I $SRC/tt_metal/hw/inc/wormhole/wormhole_b0_defines \
    -I $SRC/tt_metal/third_party/umd \
    -I $SRC/tt_metal/third_party/umd/device/wormhole \
    -I $SRC/tt_metal/third_party/umd/src/firmware/riscv/wormhole \
    -I $SRC/tt_metal/third_party/fmt \
    -I $SRC/yaml-cpp/include \
    -DTENSIX_FIRMWARE \
    -DFMT_HEADER_ONLY \
    -DFMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0 \
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
    $LIB/whisper/interp.a \
    -lpthread


