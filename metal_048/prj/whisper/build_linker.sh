#!/bin/bash

NAME=linker

SRC=../../src/whisper
LIB=../../lib/whisper

g++ -c -std=c++17 -O3 \
    -I $SRC \
    $SRC/$NAME/*.cpp

mkdir -p $LIB

ar rsc $LIB/$NAME.a *.o

rm *.o


