// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gc,
        pipe<T> pc,
        uint32 N,
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 start,
        uint32 stride) {
    pc.set_frame(frame_tiles);
    uint32 frame_items = frame_tiles * 1024;
    for (uint32 n = 0; n < N; n++) {
        uint32 pos = start;
        for (uint32 i = 0; i < num_frames; i++) {
            pc.wait_front();
            pc.write(0, gc, pos, frame_items);
            write_barrier();
            pc.pop_front();
            pos += frame_items;
        }
        start += stride;
    }
}

