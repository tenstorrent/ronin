#!/bin/bash

# SPDX-FileCopyrightText: © 2025 FRAGATA COMPUTER SYSTEMS AG
#
# SPDX-License-Identifier: Apache-2.0

NAME=vendor/arhat

CXX=/usr/lib/llvm-17/bin/clang++

SRC=../../../src
LIB=../../../lib

$CXX -c -std=c++20 -stdlib=libstdc++ -O3 \
    -I $SRC/$NAME \
    $SRC/$NAME/runtime/*.cpp

mkdir -p $LIB/$NAME

ar rsc $LIB/$NAME/runtime.a *.o

rm *.o


