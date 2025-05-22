// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank.cpp"

void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 batch,
        uint32 Mt,
        uint32 Kt,
        uint32 Nt,
        uint32 MtKt,
        uint32 KtNt,
        uint32 bcast_b) {
    constexpr uint32 onetile = 1024;
    for (uint32 nb = 0; nb < batch; nb++) {
        uint32 ga_idx = ga_pos;
        for (uint32 mt = 0; mt < Mt; mt++) {
            uint32 gb_idx = gb_pos;
            for (uint32 nt = 0; nt < Nt; nt++) {
                for (uint32 kt = 0; kt < Kt; kt++) {
                    pa.reserve_back();
                    pa.read(0, ga, ga_idx, onetile);
                    read_barrier();
                    pa.push_back();
                    pb.reserve_back();
                    pb.read(0, gb, gb_idx, onetile);
                    read_barrier();
                    pb.push_back();
                    ga_idx += onetile;
                    gb_idx += Nt * onetile;
                } // kt
                gb_idx -= KtNt * onetile;
                gb_idx += onetile;
                ga_idx -= Kt * onetile;
            } // nt
            ga_idx += Kt * onetile;
        } // mt
        ga_pos += MtKt * onetile;
        if (bcast_b == 0) {
            gb_pos += KtNt * onetile;
        }
    }
}

