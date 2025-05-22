// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pw_im,
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 Co,
        uint32 Ci,
        uint32 PQ) {
    uint32 Ct = C / 32;
    px.set_frame(Ct);
    pw.set_frame(4);
    py.set_frame(Ct);
    px_im.set_frame(Ct);
    pw_im.set_frame(4);
    pt_im.set_frame(Ci);
    py_im.set_frame(Ci);
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            // pw_im = tilize(pw)
            pw_im.reserve_back();
            pw.wait_front();
            tilize_block(pw, 4, pw_im);
            pw.pop_front();
            pw_im.push_back();
            // pw_im = transpose(pw_im)
            {
                math<T> acc;
                pw_im.wait_front();
                for (uint32 i = 0; i < 4; i++) {
                    acc.transpose(pw_im, i, i);
                }
                pw_im.pop_front();
                pw_im.reserve_back();
                for (uint32 i = 0; i < 4; i++) {
                    acc.pack(i, pw_im);
                }
                pw_im.push_back();
            } // acc
            pw_im.wait_front();
            for (uint32 i = 0; i < 4; i++) {
                // px_im = tilize(px)
                px_im.reserve_back();
                px.wait_front();
                tilize_block(px, Ct, px_im);
                px.pop_front();
                px_im.push_back();
                px_im.wait_front();
                uint32 cx = 0;
                for (uint32 co = 0; co < Co; co++) {
                    if (i == 0) {
                        // py_im = px_im * pw_im
                        math<T> acc;
                        for (uint32 ci = 0; ci < Ci; ci++) {
                            acc.mul_bcast_cols(px_im, pw_im, cx, i, ci);
                            cx++;
                        }
                        py_im.reserve_back();
                        for (uint32 ci = 0; ci < Ci; ci++) {
                            acc.pack(ci, py_im);
                        }
                        py_im.push_back();
                    } else {
                        {
                            // pt_im = px_im * pw_im
                            math<T> acc;
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.mul_bcast_cols(px_im, pw_im, cx, i, ci);
                                cx++;
                            }
                            pt_im.reserve_back();
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.pack(ci, pt_im);
                            }
                            pt_im.push_back();
                        } // acc
                        {
                            // py_im = py_im + pt_im
                            math<T> acc;
                            pt_im.wait_front();
                            py_im.wait_front();
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.add(py_im, pt_im, ci, ci, ci);
                            }
                            py_im.pop_front();
                            pt_im.pop_front();
                            py_im.reserve_back();
                            for (uint32 ci = 0; ci < Ci; ci++) {
                                acc.pack(ci, py_im);
                            }
                            py_im.push_back();
                        } // acc
                    }
                } // co
                px_im.pop_front();
            } // i
            pw_im.pop_front();
            // py = untilize(py_im)
            py_im.set_frame(Ct);
            py.reserve_back();
            py_im.wait_front();
            untilize_block(py_im, Ct, py);
            py_im.pop_front();
            py.push_back();
            py_im.set_frame(Ci);
        } // pq_start
    } // n
}

