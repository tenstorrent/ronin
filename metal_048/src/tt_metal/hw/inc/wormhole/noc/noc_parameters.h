// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "third_party/umd/src/firmware/riscv/wormhole/noc/noc_parameters.h"

#ifdef _NOC_PARAMETERS_H_

#define PCIE_NOC_X 0
#define PCIE_NOC_Y 3

#define PCIE_NOC1_X 9
#define PCIE_NOC1_Y 8

// Address formats
#define NOC_XY_ENCODING(x, y)                                        \
   (((uint32_t)(y)) << ((NOC_ADDR_LOCAL_BITS % 32)+NOC_ADDR_NODE_ID_BITS)) |  \
   (((uint32_t)(x)) << (NOC_ADDR_LOCAL_BITS % 32)) \

// Address formats
#define NOC_XY_PCIE_ENCODING(x, y, noc_index)                                        \
   NOC_XY_ENCODING(x, y) |  \
   ((noc_index ? (x == PCIE_NOC1_X and y == PCIE_NOC1_Y) : (x == PCIE_NOC_X and y == PCIE_NOC_Y)) * 0x8) \

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                \
   (((uint32_t)(x_start)) << ((NOC_ADDR_LOCAL_BITS % 32)+2*NOC_ADDR_NODE_ID_BITS)) |   \
   (((uint32_t)(y_start)) << ((NOC_ADDR_LOCAL_BITS % 32)+3*NOC_ADDR_NODE_ID_BITS)) |   \
   (((uint32_t)(x_end))   << (NOC_ADDR_LOCAL_BITS % 32)) |                             \
   (((uint32_t)(y_end))   << ((NOC_ADDR_LOCAL_BITS % 32)+NOC_ADDR_NODE_ID_BITS)) \

#define NOC_XY_ADDR2(xy, addr)                                         \
   ((((uint64_t)(xy)) << NOC_ADDR_LOCAL_BITS) |                        \
   ((uint64_t)(addr)))

// Alignment restrictions
#define NOC_L1_ALIGNMENT_BYTES   16
#define NOC_PCIE_ALIGNMENT_BYTES 32
#define NOC_DRAM_ALIGNMENT_BYTES 32

#endif
