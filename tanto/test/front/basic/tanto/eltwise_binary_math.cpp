// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

static constexpr uint32
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2;

param<uint32> eltwise_op_code;

void eltwise_op(
        math<T> acc, pipe<T> pa, pipe<T> pb, uint32 index) {
    if (eltwise_op_code == OP_ADD) {
        acc.add(pa, pb, index, index, index);
    } else if (eltwise_op_code == OP_SUB) {
        acc.sub(pa, pb, index, index, index);
    } else if (eltwise_op_code == OP_MUL) {
        acc.mul(pa, pb, index, index, index);
    }
}

void kernel(
        pipe<T> pa, 
        pipe<T> pb, 
        pipe<T> pc, 
        uint32 num_blocks,
        uint32 block_tiles) {
    for (uint32 block = 0; block < num_blocks; block++) {
        pc.reserve_back();
        pa.wait_front();
        pb.wait_front();
        math<T> acc;
        for (uint32 i = 0; i < block_tiles; i++) {
            eltwise_op(acc, pa, pb, i);
        }
        for (uint32 i = 0; i < block_tiles; i++) {
            acc.pack(i, pc);
        }
        pa.pop_front();
        pb.pop_front();
        pc.push_back();
    }
}

