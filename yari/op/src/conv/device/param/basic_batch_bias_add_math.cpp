// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;

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
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N) {
    uint32 Ct = C / 32;
    uint32 Kt = K / 32;
    px.set_frame(Ct);
    pw.set_frame(Ct);
    pb.set_frame(Kt);
    pz.set_frame(Kt);
    py.set_frame(Kt);
    px_im.set_frame(Ct);
    pz_im.set_frame(Kt);
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
            // py_im = py_im + pb
            uint32 kb = 0;
            for (uint32 ko = 0; ko < Ko; ko++) {
                math<T> acc;
                py_im.wait_front();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.add_bcast_rows(py_im, pb, ki, kb, ki);
                    kb++;
                }
                py_im.pop_front();
                py_im.reserve_back();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.pack(ki, py_im);
                }
                py_im.push_back();
            } // ko
            // pz_im = tilize(pz)
            pz_im.reserve_back();
            pz.wait_front();
            tilize_block(pz, Kt, pz_im);
            pz.pop_front();
            pz_im.push_back();
            // py_im = py_im + pz_im
            uint32 kz = 0;
            pz_im.wait_front();
            for (uint32 ko = 0; ko < Ko; ko++) {
                math<T> acc;
                py_im.wait_front();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.add(py_im, pz_im, ki, kz, ki);
                    kz++;
                }
                py_im.pop_front();
                py_im.reserve_back();
                for (uint32 ki = 0; ki < Ki; ki++) {
                    acc.pack(ki, py_im);
                }
                py_im.push_back();
            } // ko
            pz_im.pop_front();
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

