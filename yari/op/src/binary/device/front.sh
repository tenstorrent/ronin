#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL 

$FRONT --mode=read -DT=bfloat16 $TANTO/binary_batch_reader.cpp >$METAL/binary_batch_reader.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/binary_batch_writer.cpp >$METAL/binary_batch_writer.cpp

$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/binary_batch_math.cpp >$METAL/binary_batch_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/binary_batch_math.cpp >$METAL/binary_batch_sub_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=2 $TANTO/binary_batch_math.cpp >$METAL/binary_batch_mul_math.cpp

$FRONT --mode=compute -DT=bfloat16 -P0=0 -P1=0 $TANTO/binary_batch_unary_math.cpp >$METAL/binary_batch_add_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 -P1=0 $TANTO/binary_batch_unary_math.cpp >$METAL/binary_batch_sub_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=2 -P1=0 $TANTO/binary_batch_unary_math.cpp >$METAL/binary_batch_mul_relu_math.cpp

$FRONT --mode=compute -DT=bfloat16 -P0=0 -P1=1 $TANTO/binary_batch_unary_math.cpp >$METAL/binary_batch_add_relu6_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 -P1=1 $TANTO/binary_batch_unary_math.cpp >$METAL/binary_batch_sub_relu6_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=2 -P1=1 $TANTO/binary_batch_unary_math.cpp >$METAL/binary_batch_mul_relu6_math.cpp


