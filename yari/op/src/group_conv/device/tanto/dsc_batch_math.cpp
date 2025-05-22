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
        pipe<T> pa, 
        pipe<T> pb,
        uint32 pboff,
        uint32 idst,
        uint32 tiles) {
    for (uint32 i = 0; i < tiles; i++) {
        acc.matmul(pa, pb, i, i + pboff, idst, true);
    }
}

void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> pw2,
        pipe<T> pb2,
        pipe<T> py,
        pipe<T> pu_im,
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Co,
        uint32 Ci,
        uint32 Ko,
        uint32 Ki,
        uint32 KC,
        uint32 PQ,
        uint32 RS,
        uint32 RSC,
        uint32 unary_param0) {
    uint32 Ct = C / 32;
    uint32 Kt = K / 32;
    px.set_frame(Ct);
    pw.set_frame(RSC / 32);
    pb.set_frame(Ct);
    pw2.set_frame(KC / 1024);
    pb2.set_frame(Kt);
    py.set_frame(Kt);
    pu_im.set_frame(Ci);
    pt_im.set_frame(Ci);
    py_im.set_frame(Ki);
    pw.wait_front();
    pb.wait_front();
    pw2.wait_front();
    pb2.wait_front();
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            // Layer 1: depthwise
            uint32 iw = 0;
            for (uint32 rs = 0; rs < RS; rs++) {
                px.wait_front();
                uint32 cx = 0;
                for (uint32 co = 0; co < Co; co++) {
                    if (rs == 0) {
                        // pu_im = px * pw
                        math<T> acc;
                        for (uint32 ci = 0; ci < Ci; ci++) {
                            acc.mul(px, pw, cx, iw, ci);
                            cx++;
                            iw++;
                        }
                        pu_im.reserve_back();
                        for (uint32 ci = 0; ci < Ci; ci++) {
                            acc.pack(ci, pu_im);
                        }
                        pu_im.push_back();
                    } else {
                        // pt_im = px * pw
                        {
                            math<T> acc;
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.mul(px, pw, cx, iw, ci);
                                cx++;
                                iw++;
                            }
                            pt_im.reserve_back();
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.pack(ci, pt_im);
                            }
                            pt_im.push_back();
                        } // acc
                        // pu_im = pu_im + pt_im
                        {
                            math<T> acc;
                            pt_im.wait_front();
                            pu_im.wait_front();
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.add(pu_im, pt_im, ci, ci, ci);
                            }
                            pu_im.pop_front();
                            pt_im.pop_front();
                            pu_im.reserve_back();
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.pack(ci, pu_im);
                            }
                            pu_im.push_back();
                        } // acc
                    }
                } // co
                px.pop_front();
            } // rs
            // pu_im = unary(pu_im + pb)
            uint32 ib = 0;
            for (uint32 co = 0; co < Co; co++) {
                math<T> acc;
                pu_im.wait_front();
                for (uint32 ci = 0; ci < Ci; ci++) {
                    acc.add(pu_im, pb, ci, ib, ci);
                    unary_op(acc, ci, unary_param0);
                    ib++;
                }
                pu_im.pop_front();
                pu_im.reserve_back();
                for (uint32 ci = 0; ci < Ci; ci++) {
                    acc.pack(ci, pu_im);
                }
                pu_im.push_back();
            } // ko
            // Layer 2: pointwise
            // pt_im = tilize(pu_im)
            pu_im.set_frame(Ct);
            pt_im.set_frame(Ct);
            pt_im.reserve_back();
            pu_im.wait_front();
            tilize_block(pu_im, Ct, pt_im);
            pu_im.pop_front();
            pt_im.push_back();
            // py_im = matmul(pt_im, pw2) 
            pt_im.wait_front();
            uint32 iw2 = 0;
            for (uint32 ko = 0; ko < Ko; ko++) {
                math<T> acc;
                for (uint32 ki = 0; ki < Ki; ki++) {
                    matmul_slice(acc, pt_im, pw2, iw2, ki, Ct);
                    iw2 += Ct;
                }
                py_im.reserve_back();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.pack(ki, py_im);
                }
                py_im.push_back();
            } // ko
            pt_im.pop_front();
            // restore pu_im, pt_im to small frames
            pu_im.set_frame(Ci);
            pt_im.set_frame(Ci);
            // py_im = py_im + pb2
            uint32 ib2 = 0;
            for (uint32 ko = 0; ko < Ko; ko++) {
                math<T> acc;
                py_im.wait_front();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.add_bcast_rows(py_im, pb2, ki, ib2, ki);
                    ib2++;
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
    pw.pop_front();
    pb.pop_front();
    pw2.pop_front();
    pb2.pop_front();
}

