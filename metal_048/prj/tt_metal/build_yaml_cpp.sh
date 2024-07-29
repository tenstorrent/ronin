#!/bin/bash

SRC=../../src
LIB=../../lib

g++ -c -std=c++17 -O3 \
    -I $SRC/yaml-cpp/include \
    -DYAML_CPP_STATIC_DEFINE \
    $SRC/yaml-cpp/src/*.cpp

mkdir -p $LIB/tt_metal

ar rsc $LIB/tt_metal/yaml_cpp.a *.o

rm *.o


