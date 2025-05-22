#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 $TANTO/interp2d_linear_batch_reader.cpp >$METAL/interp2d_linear_batch_reader.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/interp2d_linear_batch_writer.cpp >$METAL/interp2d_linear_batch_writer.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/interp2d_linear_batch_math.cpp >$METAL/interp2d_linear_batch_math.cpp


