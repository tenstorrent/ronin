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
        uint32 pwoff,
        uint32 idst,
        uint32 tiles) {
    for (uint32 i = 0; i < tiles; i++) {
        acc.matmul(px, pw, i, i + pwoff, idst, true);
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
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 RS,
        uint32 Kb,
        uint32 RSKbC,
        uint32 unary_param0) {
    uint32 Ct = C / 32;
    uint32 Kbt = Kb / 32;
    px.set_frame(Ct);
    pw.set_frame(RSKbC / 1024);
    pb.set_frame(Kbt);
    pz.set_frame(Kbt);
    py.set_frame(Kbt);
    px_im.set_frame(Ct);
    pz_im.set_frame(Kbt);
    py_im.set_frame(Kbt);
    pw.wait_front();
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            uint32 w_start = 0;
            for (uint32 rs = 0; rs < RS; rs++) {
                // px_im = tilize(px)
                px_im.reserve_back();
                px.wait_front();
                tilize_block(px, Ct, px_im);
                px.pop_front();
                px_im.push_back();
                // py_im = matmul(px_im, pw)
                px_im.wait_front();
                math<T> acc;
                if (rs != 0) {
                    py_im.wait_front();
                    for (uint32 k = 0; k < Kbt; k++) {
                        acc.copy(py_im, k, k);
                    }
                    py_im.pop_front();
                }
                for (uint32 k = 0; k < Kbt; k++) {
                    matmul_slice(acc, px_im, pw, w_start, k, Ct);
                    w_start += Ct;
                }
                py_im.reserve_back();
                for (uint32 k = 0; k < Kbt; k++) {
                    acc.pack(k, py_im);
                }
                py_im.push_back();
                px_im.pop_front();
            } // rs
            // py_im = py_im + pb
            {
                math<T> acc;
                py_im.wait_front();
                for (uint32 k = 0; k < Kbt; k++) {
                    acc.add_bcast_rows(py_im, pb, k, k, k);
                }
                py_im.pop_front();
                py_im.reserve_back();
                for (uint32 k = 0; k < Kbt; k++) {
                    acc.pack(k, py_im);
                }
                py_im.push_back();
            } // acc
            // pz_im = tilize(pz)
            pz_im.reserve_back();
            pz.wait_front();
            tilize_block(pz, Kbt, pz_im);
            pz.pop_front();
            pz_im.push_back();
            // py_im = unary(py_im + pz_im)
            {
                math<T> acc;
                pz_im.wait_front();
                py_im.wait_front();
                for (uint32 k = 0; k < Kbt; k++) {
                    acc.add(py_im, pz_im, k, k, k);
                    unary_op(acc, k, unary_param0);
                }
                py_im.pop_front();
                py_im.reserve_back();
                for (uint32 k = 0; k < Kbt; k++) {
                    acc.pack(k, py_im);
                }
                py_im.push_back();
            } // acc
            pz_im.pop_front();
            // py = untilize(py_im)
            py.reserve_back();
            py_im.wait_front();
            untilize_block(py_im, Kbt, py);
            py_im.pop_front();
            py.push_back();
        } // pq_start
    } // n
    pw.pop_front();
}

