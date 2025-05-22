// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "bcast_w.cpp"

static constexpr uint32
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2;

param<uint32> bcast_op_code;

void bcast_op(
        math<T> acc,
        pipe<T> pa,
        pipe<T> pb) {
    if (bcast_op_code == OP_ADD) {
        acc.add_bcast_cols(pa, pb, 0, 0, 0);
    } else if (bcast_op_code == OP_SUB) {
        acc.sub_bcast_cols(pa, pb, 0, 0, 0);
    } else if (bcast_op_code == OP_MUL) {
        acc.mul_bcast_cols(pa, pb, 0, 0, 0);
    }
}

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        uint32 B,
        uint32 Ht,
        uint32 Wt) {
    for (uint32 b = 0; b < B; b++) {
        for (uint32 h = 0; h < Ht; h++) {
            pb.wait_front();
            for (uint32 w = 0; w < Wt; w++) {
                pc.reserve_back();
                math<T> acc;
                pa.wait_front();
                bcast_op(acc, pa, pb);
                acc.pack(0, pc);
                pa.pop_front();
                pc.push_back();
            }
            pb.pop_front();
        }
    }
}

