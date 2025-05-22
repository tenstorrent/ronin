// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 N,
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 start,
        uint32 stride) {
    pa.set_frame(frame_tiles);
    pb.set_frame(frame_tiles);
    uint32 frame_items = frame_tiles * 1024;
    for (uint32 n = 0; n < N; n++) {
        uint32 pos = start;
        for (uint32 i = 0; i < num_frames; i++) {
            pa.reserve_back();
            pb.reserve_back();
            pa.read(0, ga, pos, frame_items);
            pb.read(0, gb, pos, frame_items);
            read_barrier();
            pa.push_back();
            pb.push_back();
            pos += frame_items;
        }
        start += stride;
    }
}

