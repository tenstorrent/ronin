// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// originally "reader_bcast_hw_8bank.cpp"

void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 NC,
        uint32 Ht,
        uint32 Wt,
        uint32 gb_no_nc) {
    constexpr uint32 onetile = 1024;
    for (uint32 nc = 0; nc < NC; nc++) {
        for (uint32 ht = 0; ht < Ht; ht++) {
            for (uint32 wt = 0; wt < Wt; wt++) {
                pa.reserve_back();
                pa.read(0, ga, ga_pos, onetile);
                read_barrier();
                pa.push_back();
                pb.reserve_back();
                pb.read(0, gb, gb_pos, onetile);
                read_barrier();
                pb.push_back();
                ga_pos += onetile;
            }
        }
        if (gb_no_nc == 0) {
            gb_pos += onetile;
        }
    }
}

