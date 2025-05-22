// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp"

void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 Mt,
        uint32 Kt,
        uint32 Nt,
        uint32 bcast_b,
        uint32 out_tile_pos,
        uint32 out_num_tiles) {
    constexpr uint32 onetile = 1024;

    // ACHTUNG: These values can be also computed on host
    uint32 MtNt = Mt * Nt;
    uint32 KtNt = Kt * Nt;
    uint32 ga_pos = (out_tile_pos / Nt) * Kt;
    uint32 out_mtnt = out_tile_pos % MtNt;
    uint32 out_nt = out_tile_pos % Nt;
    uint32 gb_pos = out_nt;
    if (bcast_b == 0) {
        uint32 out_b = out_tile_pos / MtNt;
        gb_pos += out_b * KtNt;
    }
    ga_pos *= onetile;
    gb_pos *= onetile;

    for (uint32 n = 0; n < out_num_tiles; n++) {
        for (uint32 kt = 0; kt < Kt; kt++) {
            pa.reserve_back();
            pa.read(0, ga, ga_pos, onetile);
            read_barrier();
            pa.push_back();
            pb.reserve_back();
            pb.read(0, gb, gb_pos, onetile);
            read_barrier();
            pb.push_back();
            ga_pos += onetile;
            gb_pos += Nt * onetile;
        }
        out_mtnt++;
        out_nt++;
        gb_pos -= KtNt * onetile;
        gb_pos += onetile;
        if (out_nt == Nt) {
            out_nt = 0;
            gb_pos -= Nt * onetile;
            if (out_mtnt == MtNt) {
                out_mtnt = 0;
                if (bcast_b == 0) {
                    gb_pos += KtNt * onetile;
                }
            }
        } else {
            ga_pos -= Kt * onetile;
        }
    }
}

