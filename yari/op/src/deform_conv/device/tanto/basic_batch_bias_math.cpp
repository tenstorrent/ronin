// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

void prepare_interp(
        pipe<T> pd,
        pipe<T> pd1_im,
        pipe<T> pd2_im,
        pipe<uint16> ppi_im,
        pipe<T> ppc_im,
        uint32 Dt) {
    // pd [PQ32, D]
    pd.wait_front();
    // ppi_im = uint16(floor(pd) + 63)
    // add 63 to encode negative values as uint16
    ppi_im.reserve_back();
    for (uint32 i = 0; i < Dt; i++) {
        math<T> acc;
        constexpr uint32 C63 = 0x427c0000;
        // unpack $0
        acc.copy(pd, i, 0);
        // $0 = floor($0)
        acc.floor(0);
        // $0 = $0 + 63
        acc.fill(1, C63);
        acc.add_dst(0, 1);
        // $0 = uint16($0)
        acc.cast_bf16_u16(0);
        acc.pack(0, ppi_im);
    }
    ppi_im.push_back();
    // pd1_im = tilize(pd)
    pd1_im.reserve_back();
    tilize_block(pd, Dt, pd1_im);
    pd1_im.push_back();
    pd.pop_front();
    // pd2_im = transpose(pd1_im)
    pd2_im.reserve_back();
    pd1_im.wait_front();
    for (uint32 i = 0; i < Dt; i++) {
        math<T> acc;
        acc.transpose(pd1_im, i, 0);
        acc.pack(0, pd2_im);
    }
    pd1_im.pop_front();
    pd2_im.push_back();
    // pd1_im = tile-wise untilize (pd2_im)
    pd1_im.set_frame(1);
    pd2_im.set_frame(1);
    for (uint32 i = 0; i < Dt; i++) {
        pd1_im.reserve_back();
        pd2_im.wait_front();
        untilize_block(pd2_im, 1, pd1_im);
        pd2_im.pop_front();
        pd1_im.push_back();
    }
    pd1_im.set_frame(Dt);
    pd2_im.set_frame(Dt);
    // ppc_im = prepare(pd1_im)
    ppc_im.reserve_back();
    pd1_im.wait_front();
    for (uint32 i = 0; i < Dt; i++) {
        math<T> acc;
        // dst[2] offset
        // dst[1] lh/lw
        // dst[0] hh/hw
        constexpr uint32 ONE = 0x3f800000;
        // unpack $0
        acc.copy(pd1_im, i, 2);
        // $1 = floor($2)
        acc.copy_dst(1, 2);
        acc.floor(1);
        // $1 = $2 - $1
        acc.rsub_dst(1, 2);
        // $0 = 1.0 - $1
        acc.fill(0, ONE);
        acc.sub_dst(0, 1);
        // pack $1, $0
        acc.pack(1, ppc_im);
        acc.pack(0, ppc_im);
    }
    pd1_im.pop_front();
    ppc_im.push_back();
}

void interp(
        pipe<T> px,
        pipe<T> px_im,
        pipe<T> pt_im,
        pipe<T> pc1_im,
        pipe<T> pc2_im,
        uint32 Ct,
        uint32 Co,
        uint32 Ci) {
    // pc2_im = tilize(pc1_im)
    pc2_im.reserve_back();
    pc1_im.wait_front();
    tilize_block(pc1_im, 4, pc2_im);
    pc1_im.pop_front();
    pc2_im.push_back();
    // pc2_im = transpose(pc2_im)
    {
        math<T> acc;
        pc2_im.wait_front();
        for (uint32 i = 0; i < 4; i++) {
            acc.transpose(pc2_im, i, i);
        }
        pc2_im.pop_front();
        pc2_im.reserve_back();
        for (uint32 i = 0; i < 4; i++) {
            acc.pack(i, pc2_im);
        }
        pc2_im.push_back();
    }
    pc2_im.wait_front();
    // px_im = 0
    px_im.reserve_back();
    for (uint32 c = 0; c < Ct; c++) {
        math<T> acc;
        acc.fill(0, 0);
        acc.pack(0, px_im);
    }
    px_im.push_back();
    for (uint32 i = 0; i < 4; i++) {
        // pt_im = tilize(px)
        pt_im.reserve_back();
        px.wait_front();
        tilize_block(px, Ct, pt_im);
        px.pop_front();
        pt_im.push_back();
        // c0 = hh * hw
        // c1 = hh * lw
        // c2 = lh * hw
        // c3 = lh * lw
        // [th, tw] in [[0, 2], [0, 3], [1, 2], [1, 3]]
        //     where 0 = hh, 1 = lh, 2 = hw, 3 = lw
        uint32 th = (i >> 1) & 1;
        uint32 tw = (i & 1) + 2;
        // narrow frames
        px_im.set_frame(Ci);
        pt_im.set_frame(Ci);
        // pt_im = pt_im * pc2_im[th]
        for (uint32 co = 0; co < Co; co++) {
            math<T> acc;
            pt_im.wait_front();
            for (uint32 ci = 0; ci < Ci; ci++) {
                acc.mul_bcast_cols(pt_im, pc2_im, ci, th, ci);
            }
            pt_im.pop_front();
            pt_im.reserve_back();
            for (uint32 ci = 0; ci < Ci; ci++) {
                acc.pack(ci, pt_im);
            }
            pt_im.push_back();
        }
        // pt_im = pt_im * pc2_im[tw]
        for (uint32 co = 0; co < Co; co++) {
            math<T> acc;
            pt_im.wait_front();
            for (uint32 ci = 0; ci < Ci; ci++) {
                acc.mul_bcast_cols(pt_im, pc2_im, ci, tw, ci);
            }
            pt_im.pop_front();
            pt_im.reserve_back();
            for (uint32 ci = 0; ci < Ci; ci++) {
                acc.pack(ci, pt_im);
            }
            pt_im.push_back();
        }
        // px_im = px_im + pt_im
        for (uint32 co = 0; co < Co; co++) {
            math<T> acc;
            px_im.wait_front();
            pt_im.wait_front();
            for (uint32 ci = 0; ci < Ci; ci++) {
                acc.add(px_im, pt_im, ci, ci, ci);
            }
            px_im.pop_front();
            pt_im.pop_front();
            px_im.reserve_back();
            for (uint32 ci = 0; ci < Ci; ci++) {
                acc.pack(ci, px_im);
            }
            px_im.push_back();
        }
        // restore frames
        px_im.set_frame(Ct);
        pt_im.set_frame(Ct);
    }
    pc2_im.pop_front();
}

void kernel(
        pipe<T> px,
        pipe<T> pd,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pt_im,
        pipe<T> py_im,
        pipe<T> pd1_im,
        pipe<T> pd2_im,
        pipe<uint16> ppi_im,
        pipe<T> ppc_im,
        pipe<T> pc1_im,
        pipe<T> pc2_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 D,
        uint32 Co,
        uint32 Ci,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS) {
    uint32 Ct = C / 32;
    uint32 Kt = K / 32;
    uint32 Dt = D / 32;
    px.set_frame(Ct);
    pd.set_frame(Dt);
    pw.set_frame(Ct);
    pb.set_frame(Kt);
    py.set_frame(Kt);
    px_im.set_frame(Ct);
    pt_im.set_frame(Ct);
    py_im.set_frame(Ki);
    pd1_im.set_frame(Dt);
    pd2_im.set_frame(Dt);
    ppi_im.set_frame(Dt);
    ppc_im.set_frame(Dt * 2);
    pc1_im.set_frame(4);
    pc2_im.set_frame(4);
    pb.wait_front();
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            prepare_interp(pd, pd1_im, pd2_im, ppi_im, ppc_im, Dt);
            for (uint32 rs = 0; rs < RS; rs++) {
                // px_im = interp(px)
                interp(px, px_im, pt_im, pc1_im, pc2_im, Ct, Co, Ci);
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

