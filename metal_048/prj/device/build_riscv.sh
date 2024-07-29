#!/bin/bash

NAME=riscv

SRC=../../src
LIB=../../lib

g++ -c -std=c++17 -O3 \
    -I $SRC \
    -I $SRC/device \
    $SRC/device/$NAME/*.cpp

mkdir -p $LIB/device

ar rsc $LIB/device/${NAME}.a *.o

rm *.o


