// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank.cpp"

void kernel(
        global<T> gc,
        pipe<T> pc,
        uint32 gc_pos,
        uint32 batch,
        uint32 Mt,
        uint32 Nt) {
    constexpr uint32 onetile = 1024;
    for (uint32 nb = 0; nb < batch; nb++) {
        for (uint32 mt = 0; mt < Mt; mt++) {
            for (uint32 nt = 0; nt < Nt; nt++) {
                pc.wait_front();
                pc.write(0, gc, gc_pos, onetile);
                write_barrier();
                pc.pop_front();
                gc_pos += onetile;
            }
        }
    }
}

