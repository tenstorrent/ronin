#!/bin/bash

SRC=../../src
LIB=../../lib

g++ -c -std=c++17 -O3 \
    -I $SRC \
    -I $SRC/tt_metal \
    -I $SRC/tt_metal/hw/inc \
    -I $SRC/tt_metal/emulator/hw \
    -I $SRC/tt_metal/third_party/umd \
    -I $SRC/tt_metal/third_party/fmt \
    -I $SRC/yaml-cpp/include \
    -DTT_METAL_EMULATOR \
    -DTENSIX_FIRMWARE \
    -DFMT_HEADER_ONLY \
    -DFMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0 \
    -DYAML_CPP_STATIC_DEFINE \
    $SRC/tt_metal/llrt/*.cpp

mkdir -p $LIB/tt_metal

ar rsc $LIB/tt_metal/llrt.a *.o

rm *.o


