#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

./build_api.sh
./build_arch.sh
./build_core.sh
./build_dispatch.sh
./build_ref.sh
./build_riscv.sh
./build_schedule.sh


