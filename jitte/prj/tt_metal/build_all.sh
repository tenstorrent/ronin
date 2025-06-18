#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

./build_common.sh
./build_device.sh
./build_emulator.sh
./build_jit_build.sh
./build_llrt.sh
./build_tt_metal.sh
./build_tt_metal_detail.sh
./build_tt_metal_impl.sh
./build_yaml_cpp.sh


