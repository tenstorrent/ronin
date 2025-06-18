#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

export JITTE_HOME=../../../jitte/home

mkdir -p $JITTE_HOME/algo/basic

cp -R -v ../../src/basic/device $JITTE_HOME/algo/basic


