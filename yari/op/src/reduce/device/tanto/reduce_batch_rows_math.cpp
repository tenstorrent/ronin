// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

static constexpr uint32
    REDUCE_OP_MAX = 0,
    REDUCE_OP_SUM = 1;

param<uint32> reduce_op_code;

void reduce_op(math<T> acc, pipe<T> px, pipe<T> ps) {
    if (reduce_op_code == REDUCE_OP_MAX) {
        acc.reduce_max_rows(px, ps, 0, 0, 0);
    } else if (reduce_op_code == REDUCE_OP_SUM) {
        acc.reduce_sum_rows(px, ps, 0, 0, 0);
    }
}

void kernel(
        pipe<T> px,
        pipe<T> ps,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 H,
        uint32 W) {
    uint32 Ht = H / 32;
    uint32 Wt = W / 32;
    px.set_frame(Wt);
    ps.set_frame(1);
    py.set_frame(1);
    px_im.set_frame(Wt);
    py_im.set_frame(1);
    ps.wait_front();
    for (uint32 n = 0; n < N; n++) {
        for (uint32 h = 0; h < Ht; h++) {
            px_im.reserve_back();
            px.wait_front();
            tilize_block(px, Wt, px_im);
            px.pop_front();
            px_im.push_back();
            px_im.set_frame(1);
            {
                math<T> acc;
                for (uint32 w = 0; w < Wt; w++) {
                    px_im.wait_front();
                    reduce_op(acc, px_im, ps);
                    px_im.pop_front();
                }
                py_im.reserve_back();
                acc.pack_row(0, py_im);
                py_im.push_back();
            }
            px_im.set_frame(Wt);
            py.reserve_back();
            py_im.wait_front();
            untilize_block(py_im, 1, py);
            py_im.pop_front();
            py.push_back();
        }
    }
    ps.pop_front();
}

