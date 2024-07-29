#!/bin/bash

./build_eltwise_binary.sh
./build_eltwise_sfpu.sh
./build_loopback.sh
./build_matmul_multicore_reuse_mcast.sh
./build_matmul_multicore_reuse.sh
./build_matmul_multi_core.sh
./build_matmul_single_core.sh

