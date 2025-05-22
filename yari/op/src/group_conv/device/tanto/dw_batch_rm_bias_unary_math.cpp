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

void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSK,
        uint32 unary_param0) {
    uint32 Kt = K / 32;
    px.set_frame(Kt);
    pw.set_frame(RSK / 32);
    pb.set_frame(Kt);
    py.set_frame(Ki);
    pt_im.set_frame(Ki);
    py_im.set_frame(Ki);
    pw.wait_front();
    pb.wait_front();
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            uint32 iw = 0;
            for (uint32 rs = 0; rs < RS; rs++) {
                px.wait_front();
                uint32 kx = 0;
                for (uint32 ko = 0; ko < Ko; ko++) {
                    if (rs == 0) {
                        math<T> acc;
                        for (uint32 ki = 0; ki < Ki; ki++) {
                            acc.mul(px, pw, kx, iw, ki);
                            kx++;
                            iw++;
                        }
                        py_im.reserve_back();
                        for (uint32 ki = 0; ki < Ki; ki++) {
                            acc.pack(ki, py_im);
                        }
                        py_im.push_back();
                    } else {
                        {
                            math<T> acc;
                            for (uint32 ki = 0; ki < Ki; ki++) {
                                acc.mul(px, pw, kx, iw, ki);
                                kx++;
                                iw++;
                            }
                            pt_im.reserve_back();
                            for (uint32 ki = 0; ki < Ki; ki++) {
                                acc.pack(ki, pt_im);
                            }
                            pt_im.push_back();
                        } // acc
                        {
                            math<T> acc;
                            pt_im.wait_front();
                            py_im.wait_front();
                            for (uint32 ki = 0; ki < Ki; ki++) {
                                acc.add(py_im, pt_im, ki, ki, ki);
                            }
                            py_im.pop_front();
                            pt_im.pop_front();
                            py_im.reserve_back();
                            for (uint32 ki = 0; ki < Ki; ki++) {
                                acc.pack(ki, py_im);
                            }
                            py_im.push_back();
                        } // acc
                    }
                } // ko
                px.pop_front();
            } // rs
            uint32 kb = 0;
            for (uint32 ko = 0; ko < Ko; ko++) {
                math<T> acc;
                py_im.wait_front();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.add(py_im, pb, ki, kb, ki);
                    unary_op(acc, ki, unary_param0);
                    kb++;
                }
                py_im.pop_front();
                py.reserve_back();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.pack(ki, py);
                }
                py.push_back();
            } // ko
        } // pq_start
    } // n
    pb.pop_front();
    pw.pop_front();
}

