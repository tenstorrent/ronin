// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

static constexpr uint32
    UNARY_OP_RELU = 0,
    UNARY_OP_RELU6 = 1;

param<uint32> unary_op_code;

void unary_op(math<T> acc, uint32 index, uint32 param0) {
    if (unary_op_code == UNARY_OP_RELU) {
        acc.relu(index);
    } else if (unary_op_code == UNARY_OP_RELU6) {
        acc.relu_max(index, 0x40c00000);
    }
}

void matmul_slice(
        math<T> acc, 
        pipe<T> px, 
        pipe<T> pw, 
        uint32 idst,
        uint32 tiles) {
    for (uint32 i = 0; i < tiles; i++) {
        acc.matmul(px, pw, i, i, idst, true);
    }
}

void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 unary_param0) {
    uint32 Ct = C / 32;
    uint32 Kt = K / 32;
    px.set_frame(Ct);
    pw.set_frame(Ct);
    pb.set_frame(Kt);
    py.set_frame(Kt);
    px_im.set_frame(Ct);
    py_im.set_frame(Ki);
    pb.wait_front();
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            for (uint32 rs = 0; rs < RS; rs++) {
                // px_im = tilize(px)
                px_im.reserve_back();
                px.wait_front();
                tilize_block(px, Ct, px_im);
                px.pop_front();
                px_im.push_back();
                // py_im = matmul(px_im, pw)
                px_im.wait_front();
                for (uint32 ko = 0; ko < Ko; ko++) {
                    math<T> acc;
                    if (rs != 0) {
                        py_im.wait_front();
                        for (uint32 ki = 0; ki < Ki; ki++) {
                            acc.copy(py_im, ki, ki);
                        }
                        py_im.pop_front();
                    }
                    for (uint32 ki = 0; ki < Ki; ki++) {
                        pw.wait_front();
                        matmul_slice(acc, px_im, pw, ki, Ct);
                        pw.pop_front();
                    }
                    py_im.reserve_back();
                    for (uint32 ki = 0; ki < Ki; ki++) {
                        acc.pack(ki, py_im);
                    }
                    py_im.push_back();
                } // ko
                px_im.pop_front();
            } // rs
            // py_im = unary(py_im + pb)
            uint32 kb = 0;
            for (uint32 ko = 0; ko < Ko; ko++) {
                math<T> acc;
                py_im.wait_front();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.add_bcast_rows(py_im, pb, ki, kb, ki);
                    unary_op(acc, ki, unary_param0);
                    kb++;
                }
                py_im.pop_front();
                py_im.reserve_back();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.pack(ki, py_im);
                }
                py_im.push_back();
            } // ko
            // py = untilize(py_im)
            py_im.set_frame(Kt);
            py.reserve_back();
            py_im.wait_front();
            untilize_block(py_im, Kt, py);
            py_im.pop_front();
            py.push_back();
            py_im.set_frame(Ki);
        } // pq_start
    } // n
    pb.pop_front();
}

