// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "reduce_h.cpp"

static constexpr uint32
    REDUCE_OP_MAX = 0,
    REDUCE_OP_SUM = 1;

param<uint32> reduce_op_code;

void reduce_op(math<T> acc, pipe<T> px, pipe<T> ps) {
    if (reduce_op_code == REDUCE_OP_MAX) {
        acc.reduce_max_cols(px, ps, 0, 0, 0);
    } else if (reduce_op_code == REDUCE_OP_SUM) {
        acc.reduce_sum_cols(px, ps, 0, 0, 0);
    }
}

void kernel(
        pipe<T> px,
        pipe<T> ps,
        pipe<T> py,
        uint32 Ht,
        uint32 Wt,
        uint32 NC) {
    ps.wait_front();
    for (uint32 nc = 0; nc < NC; nc++) {
        for (uint32 wt = 0; wt < Wt; wt++) {
            // tiles are expected to be coming in NCHW order (H-contiguous)
            math<T> acc;
            for (uint32 ht = 0; ht < Ht; ht++) {
                px.wait_front();
                reduce_op(acc, px, ps);
                px.pop_front();
            }
            py.reserve_back();
            acc.pack_col(0, py);
            py.push_back();
        }
    }
}

