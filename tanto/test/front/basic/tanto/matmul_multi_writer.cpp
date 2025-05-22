// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_unary_interleaved_start_id.cpp"

void kernel(
        global<T> gc,
        pipe<T> pc,
        uint32 tile_pos,
        uint32 num_tiles) {
    constexpr uint32 onetile = 1024;
    uint32 gc_pos = tile_pos * onetile;
    for (uint32 i = 0; i < num_tiles; i++) {
        pc.wait_front();
        pc.write(0, gc, gc_pos, onetile);
        write_barrier();
        pc.pop_front();
        gc_pos += onetile;
    }
}

