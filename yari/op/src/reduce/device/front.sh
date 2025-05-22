#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 $TANTO/reduce_batch_reader.cpp >$METAL/reduce_batch_reader.cpp

$FRONT --mode=write -DT=bfloat16 $TANTO/reduce_batch_cols_writer.cpp >$METAL/reduce_batch_cols_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/reduce_batch_rows_writer.cpp >$METAL/reduce_batch_rows_writer.cpp

$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/reduce_batch_cols_math.cpp >$METAL/reduce_batch_max_cols_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/reduce_batch_cols_math.cpp >$METAL/reduce_batch_sum_cols_math.cpp

$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/reduce_batch_rows_math.cpp >$METAL/reduce_batch_max_rows_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/reduce_batch_rows_math.cpp >$METAL/reduce_batch_sum_rows_math.cpp


