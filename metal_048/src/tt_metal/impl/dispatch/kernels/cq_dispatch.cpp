// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch kernel
//  - receives data in pages from prefetch kernel into the dispatch buffer ring buffer
//  - processes commands with embedded data from the dispatch buffer to write/sync/etc w/ destination
//  - sync w/ prefetcher is via 2 semaphores, page_ready, page_done
//  - page size must be a power of 2
//  - # blocks must evenly divide the dispatch buffer size
//  - dispatch buffer base must be page size aligned

#include "debug/assert.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"

// The command queue write interface controls writes to the completion region, host owns the completion region read
// interface Data requests from device and event states are written to the completion region

CQWriteInterface cq_write_interface;

constexpr uint32_t dispatch_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t dispatch_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t my_dispatch_cb_sem_id = get_compile_time_arg_val(3);
constexpr uint32_t upstream_dispatch_cb_sem_id = get_compile_time_arg_val(4);
constexpr uint32_t dispatch_cb_blocks = get_compile_time_arg_val(5);
constexpr uint32_t upstream_sync_sem = get_compile_time_arg_val(6);
constexpr uint32_t command_queue_base_addr = get_compile_time_arg_val(7);
constexpr uint32_t completion_queue_base_addr = get_compile_time_arg_val(8);
constexpr uint32_t completion_queue_size = get_compile_time_arg_val(9);
constexpr uint32_t downstream_cb_base = get_compile_time_arg_val(10);
constexpr uint32_t downstream_cb_size = get_compile_time_arg_val(11);
constexpr uint32_t my_downstream_cb_sem_id = get_compile_time_arg_val(12);
constexpr uint32_t downstream_cb_sem_id = get_compile_time_arg_val(13);
constexpr uint32_t split_dispatch_page_preamble_size = get_compile_time_arg_val(14);
constexpr uint32_t is_d_variant = get_compile_time_arg_val(15);
constexpr uint32_t is_h_variant = get_compile_time_arg_val(16);

constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t pcie_noc_xy = uint32_t(NOC_XY_PCIE_ENCODING(NOC_0_X(static_cast<uint8_t>(NOC_INDEX), noc_size_x, PCIE_NOC_X), NOC_0_Y(static_cast<uint8_t>(NOC_INDEX), noc_size_y, PCIE_NOC_Y), NOC_INDEX));
constexpr uint32_t dispatch_cb_page_size = 1 << dispatch_cb_log_page_size;

constexpr uint32_t completion_queue_end_addr = completion_queue_base_addr + completion_queue_size;
constexpr uint32_t completion_queue_page_size = dispatch_cb_page_size;
constexpr uint32_t completion_queue_log_page_size = dispatch_cb_log_page_size;
constexpr uint32_t completion_queue_size_16B = completion_queue_size >> 4;
constexpr uint32_t completion_queue_page_size_16B = completion_queue_page_size >> 4;
constexpr uint32_t completion_queue_end_addr_16B = completion_queue_end_addr >> 4;
constexpr uint32_t completion_queue_base_addr_16B = completion_queue_base_addr >> 4;
constexpr uint32_t dispatch_cb_size = dispatch_cb_page_size * dispatch_cb_pages;
constexpr uint32_t dispatch_cb_end = dispatch_cb_base + dispatch_cb_size;
constexpr uint32_t downstream_cb_end = downstream_cb_base + downstream_cb_size;

// Break buffer into blocks, 1/n of the total (dividing equally)
// Do bookkeeping (release, etc) based on blocks
// Note: due to the current method of release pages, up to 1 block of pages
// may be unavailable to the prefetcher at any time
constexpr uint32_t dispatch_cb_pages_per_block = dispatch_cb_pages / dispatch_cb_blocks;

static uint32_t block_next_start_addr[dispatch_cb_blocks];
static uint32_t block_noc_writes_to_clear[dispatch_cb_blocks];
static uint32_t rd_block_idx;
static uint32_t wr_block_idx;

static uint32_t cb_fence;  // walks through cb page by page
static uint32_t cmd_ptr;   // walks through pages in cb cmd by cmd

static uint32_t downstream_cb_data_ptr = downstream_cb_base;

constexpr uint32_t l1_to_local_cache_copy_chunk = 6;
constexpr uint32_t max_write_packed_cores =
    108;  // GS 120 - 1 row TODO: this should be a compile time arg passed in from host
constexpr uint32_t l1_cache_size =
    ((max_write_packed_cores + l1_to_local_cache_copy_chunk - 1) / l1_to_local_cache_copy_chunk) *
    l1_to_local_cache_copy_chunk;

static uint32_t l1_cache[l1_cache_size];

// NOTE: CAREFUL USING THIS FUNCTION
// It is call "careful_copy" because you need to be careful...
// It copies beyond count by up to 5 elements make sure src and dst addresses are safe
FORCE_INLINE
void careful_copy_from_l1_to_local_cache(volatile uint32_t tt_l1_ptr *l1_ptr, uint32_t count) {
    uint32_t n = 0;
    ASSERT(l1_to_local_cache_copy_chunk == 6);
    ASSERT(count <= l1_cache_size);
    while (n < count) {
        uint32_t v0 = l1_ptr[n + 0];
        uint32_t v1 = l1_ptr[n + 1];
        uint32_t v2 = l1_ptr[n + 2];
        uint32_t v3 = l1_ptr[n + 3];
        uint32_t v4 = l1_ptr[n + 4];
        uint32_t v5 = l1_ptr[n + 5];
        l1_cache[n + 0] = v0;
        l1_cache[n + 1] = v1;
        l1_cache[n + 2] = v2;
        l1_cache[n + 3] = v3;
        l1_cache[n + 4] = v4;
        l1_cache[n + 5] = v5;
        n += 6;
    }
}

FORCE_INLINE volatile uint32_t *get_cq_completion_read_ptr() {
    return reinterpret_cast<volatile uint32_t *>(CQ_COMPLETION_READ_PTR);
}

FORCE_INLINE volatile uint32_t *get_cq_completion_write_ptr() {
    return reinterpret_cast<volatile uint32_t *>(CQ_COMPLETION_WRITE_PTR);
}

FORCE_INLINE
void completion_queue_reserve_back(uint32_t num_pages) {
    DEBUG_STATUS("QRBW");
    // Transfer pages are aligned
    uint32_t data_size_16B = num_pages * completion_queue_page_size_16B;
    uint32_t completion_rd_ptr_and_toggle;
    uint32_t completion_rd_ptr;
    uint32_t completion_rd_toggle;
    uint32_t available_space;
    do {
        completion_rd_ptr_and_toggle = *get_cq_completion_read_ptr();
        completion_rd_ptr = completion_rd_ptr_and_toggle & 0x7fffffff;
        completion_rd_toggle = completion_rd_ptr_and_toggle >> 31;
        // Toggles not equal means write ptr has wrapped but read ptr has not
        // so available space is distance from write ptr to read ptr
        // Toggles are equal means write ptr is ahead of read ptr
        // so available space is total space minus the distance from read to write ptr
        available_space =
            completion_rd_toggle != cq_write_interface.completion_fifo_wr_toggle
                ? completion_rd_ptr - cq_write_interface.completion_fifo_wr_ptr
                : (completion_queue_size_16B - (cq_write_interface.completion_fifo_wr_ptr - completion_rd_ptr));
    } while (data_size_16B > available_space);

    DEBUG_STATUS("QRBD");
}

FORCE_INLINE
void notify_host_of_completion_queue_write_pointer() {
    uint64_t completion_queue_write_ptr_addr = command_queue_base_addr + HOST_CQ_COMPLETION_WRITE_PTR;
    uint64_t pcie_address = get_noc_addr_helper(pcie_noc_xy, completion_queue_write_ptr_addr);  // For now, we are writing to host hugepages at offset
    uint32_t completion_wr_ptr_and_toggle = cq_write_interface.completion_fifo_wr_ptr | (cq_write_interface.completion_fifo_wr_toggle << 31);
    volatile tt_l1_ptr uint32_t* completion_wr_ptr_addr = get_cq_completion_write_ptr();
    completion_wr_ptr_addr[0] = completion_wr_ptr_and_toggle;
    noc_async_write_one_packet(CQ_COMPLETION_WRITE_PTR, pcie_address, 4);
    block_noc_writes_to_clear[rd_block_idx]++;
}

FORCE_INLINE
void completion_queue_push_back(uint32_t num_pages) {
    // Transfer pages are aligned
    uint32_t push_size_16B = num_pages * completion_queue_page_size_16B;
    cq_write_interface.completion_fifo_wr_ptr += push_size_16B;

    if (cq_write_interface.completion_fifo_wr_ptr >= completion_queue_end_addr_16B) {
        cq_write_interface.completion_fifo_wr_ptr =
            cq_write_interface.completion_fifo_wr_ptr - completion_queue_end_addr_16B + completion_queue_base_addr_16B;
        // Flip the toggle
        cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
    }

    // Notify host of updated completion wr ptr
    notify_host_of_completion_queue_write_pointer();
}

void process_write_host_h() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;

    uint32_t completion_write_ptr;
    // We will send the cmd back in the first X bytes, this makes the logic of reserving/pushing completion queue
    // pages much simpler since we are always sending writing full pages (except for last page)
    uint32_t length = cmd->write_linear_host.length;
    DPRINT << "process_write_host_h: " << length << ENDL();
    uint32_t data_ptr = cmd_ptr;
    while (length != 0) {
        // Get a page if needed
        if (cb_fence == data_ptr) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }
                move_rd_to_next_block<dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }
            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_noc_xy, my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx);
            ;
            cb_fence += n_pages * dispatch_cb_page_size;

            // Release pages for prefetcher
            // Since we gate how much we acquire to < 1/4 the buffer, this should be called enough
            cb_block_release_pages<
                upstream_noc_xy,
                upstream_dispatch_cb_sem_id,
                dispatch_cb_blocks,
                dispatch_cb_pages_per_block>(block_noc_writes_to_clear, wr_block_idx);
        }
        uint32_t available_data = cb_fence - data_ptr;
        uint32_t xfer_size = (length > available_data) ? available_data : length;
        uint32_t npages = (xfer_size + completion_queue_page_size - 1) / completion_queue_page_size;
        completion_queue_reserve_back(npages);
        uint32_t completion_queue_write_addr = cq_write_interface.completion_fifo_wr_ptr << 4;
        uint64_t host_completion_queue_write_addr = get_noc_addr_helper(pcie_noc_xy, completion_queue_write_addr);
        // completion_queue_write_addr will never be equal to completion_queue_end_addr due to completion_queue_push_back
        // wrap logic so we don't need to handle this case explicitly to avoid 0 sized transactions
        if (completion_queue_write_addr + xfer_size > completion_queue_end_addr) {
            uint32_t last_chunk_size = completion_queue_end_addr - completion_queue_write_addr;
            noc_async_write(data_ptr, host_completion_queue_write_addr, last_chunk_size);
            completion_queue_write_addr = completion_queue_base_addr;
            data_ptr += last_chunk_size;
            length -= last_chunk_size;
            xfer_size -= last_chunk_size;
            host_completion_queue_write_addr = get_noc_addr_helper(pcie_noc_xy, completion_queue_write_addr);
            block_noc_writes_to_clear[rd_block_idx]+=(last_chunk_size + NOC_MAX_BURST_SIZE - 1) / NOC_MAX_BURST_SIZE; // XXXXX maybe just write the noc internal api counter
        }
        noc_async_write(data_ptr, host_completion_queue_write_addr, xfer_size);
        // This will update the write ptr on device and host
        // We flush to ensure the ptr has been read out of l1 before we update it again
        completion_queue_push_back(npages);
        noc_async_writes_flushed();
        block_noc_writes_to_clear[rd_block_idx] +=
            (xfer_size + NOC_MAX_BURST_SIZE - 1) /
            NOC_MAX_BURST_SIZE;  // XXXXX maybe just write the noc internal api counter

        length -= xfer_size;
        data_ptr += xfer_size;
    }
    cmd_ptr = data_ptr;
}

// Relay, potentially through the mux/dmux/tunneller path
// Code below sends 1 page worth of data except at the end of a cmd
// This means the downstream buffers are always page aligned, simplifies wrap handling
template <uint32_t preamble_size>
void relay_to_next_cb(uint32_t data_ptr, uint32_t length) {
    static_assert(
        preamble_size == 0 || preamble_size == sizeof(dispatch_packet_header_t),
        "Dispatcher preamble size must be 0 or sizeof(dispatch_packet_header_t)");

    DPRINT << "relay_to_next_cb: " << data_ptr << " " << cb_fence << " " << length << ENDL();

    // First page should be valid since it has the command
    ASSERT(data_ptr <= dispatch_cb_end - dispatch_cb_page_size);
    ASSERT(data_ptr <= cb_fence - dispatch_cb_page_size);

    while (length > 0) {
        ASSERT(downstream_cb_end > downstream_cb_data_ptr);

        cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(1);

        uint32_t xfer_size;
        bool not_end_of_cmd;
        if (length > dispatch_cb_page_size - preamble_size) {
            xfer_size = dispatch_cb_page_size - preamble_size;
            not_end_of_cmd = true;
        } else {
            xfer_size = length;
            not_end_of_cmd = false;
        }

        uint64_t dst = get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr);

        if (preamble_size > 0) {
            uint32_t flag;
            noc_inline_dw_write(dst, xfer_size + preamble_size + not_end_of_cmd);
            block_noc_writes_to_clear[rd_block_idx]++;
            downstream_cb_data_ptr += preamble_size;
            dst = get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr);
            ASSERT(downstream_cb_data_ptr < downstream_cb_end);
        }

        // Get a page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    ASSERT(cb_fence == dispatch_cb_end);
                    uint32_t orphan_size = cb_fence - data_ptr;
                    if (orphan_size != 0) {
                        noc_async_write<dispatch_cb_page_size>(data_ptr, dst, orphan_size);
                        block_noc_writes_to_clear[rd_block_idx]++;
                        length -= orphan_size;
                        xfer_size -= orphan_size;
                        downstream_cb_data_ptr += orphan_size;
                        if (downstream_cb_data_ptr == downstream_cb_end) {
                            downstream_cb_data_ptr = downstream_cb_base;
                        }
                        dst = get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr);
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }

                move_rd_to_next_block<dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_noc_xy, my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx);
            cb_fence += n_pages * dispatch_cb_page_size;

            // Release pages for prefetcher
            // Since we gate how much we acquire to < 1/4 the buffer, this should be called enough
            cb_block_release_pages<
                upstream_noc_xy,
                upstream_dispatch_cb_sem_id,
                dispatch_cb_blocks,
                dispatch_cb_pages_per_block>(block_noc_writes_to_clear, wr_block_idx);
        }

        noc_async_write<dispatch_cb_page_size>(data_ptr, dst, xfer_size);
        block_noc_writes_to_clear[rd_block_idx]++;  // XXXXX maybe just write the noc internal api counter
        cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(1);  // XXXX optimize, take all available

        length -= xfer_size;
        data_ptr += xfer_size;
        downstream_cb_data_ptr += xfer_size;
        if (downstream_cb_data_ptr == downstream_cb_end) {
            downstream_cb_data_ptr = downstream_cb_base;
        }
    }

    // Move to next page
    downstream_cb_data_ptr = round_up_pow2(downstream_cb_data_ptr, dispatch_cb_page_size);
    if (downstream_cb_data_ptr == downstream_cb_end) {
        downstream_cb_data_ptr = downstream_cb_base;
    }

    cmd_ptr = data_ptr;
}

void process_write_host_d() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;
    // Remember: host transfer command includes the command in the payload, don't add it here
    uint32_t length = cmd->write_linear_host.length;
    uint32_t data_ptr = cmd_ptr;

    relay_to_next_cb<split_dispatch_page_preamble_size>(data_ptr, length);
}

void relay_write_h() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;
    uint32_t length = sizeof(CQDispatchCmd) + cmd->write_linear.length;
    uint32_t data_ptr = cmd_ptr;

    relay_to_next_cb<split_dispatch_page_preamble_size>(data_ptr, length);
}

// Note that for non-paged writes, the number of writes per page is always 1
// This means each noc_write frees up a page
template <bool multicast>
void process_write_linear(uint32_t num_mcast_dests) {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;

    uint32_t dst_noc = cmd->write_linear.noc_xy_addr;
    uint32_t dst_addr = cmd->write_linear.addr;
    uint32_t length = cmd->write_linear.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    while (length != 0) {
        uint32_t xfer_size = (length > dispatch_cb_page_size) ? dispatch_cb_page_size : length;
        uint64_t dst = get_noc_addr_helper(dst_noc, dst_addr);

        // Get a page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    uint32_t orphan_size = dispatch_cb_end - data_ptr;
                    if (orphan_size != 0) {
                        if constexpr (multicast) {
                            noc_async_write_multicast<dispatch_cb_page_size>(
                                data_ptr, dst, orphan_size, num_mcast_dests);
                        } else {
                            noc_async_write<dispatch_cb_page_size>(data_ptr, dst, orphan_size);
                        }
                        block_noc_writes_to_clear[rd_block_idx]++;
                        length -= orphan_size;
                        xfer_size -= orphan_size;
                        dst_addr += orphan_size;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                    dst = get_noc_addr_helper(dst_noc, dst_addr);
                }

                move_rd_to_next_block<dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_noc_xy, my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx);
            cb_fence += n_pages * dispatch_cb_page_size;

            // Release pages for prefetcher
            // Since we gate how much we acquire to < 1/4 the buffer, this should be called enough
            cb_block_release_pages<
                upstream_noc_xy,
                upstream_dispatch_cb_sem_id,
                dispatch_cb_blocks,
                dispatch_cb_pages_per_block>(block_noc_writes_to_clear, wr_block_idx);
        }

        if constexpr (multicast) {
            noc_async_write_multicast<dispatch_cb_page_size>(data_ptr, dst, xfer_size, num_mcast_dests);
        } else {
            noc_async_write<dispatch_cb_page_size>(data_ptr, dst, xfer_size);
        }
        block_noc_writes_to_clear[rd_block_idx]++;  // XXXXX maybe just write the noc internal api counter

        length -= xfer_size;
        data_ptr += xfer_size;
        dst_addr += xfer_size;
    }
    cmd_ptr = data_ptr;
}

void process_write() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;
    uint32_t num_mcast_dests = cmd->write_linear.num_mcast_dests;
    if (num_mcast_dests == 0) {
        process_write_linear<false>(0);
    } else {
        process_write_linear<true>(num_mcast_dests);
    }
}

template <bool is_dram>
void process_write_paged() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;

    uint32_t page_id = cmd->write_paged.start_page;
    uint32_t base_addr = cmd->write_paged.base_addr;
    uint32_t page_size = cmd->write_paged.page_size;
    uint32_t pages = cmd->write_paged.pages;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    uint32_t write_length = pages * page_size;
    InterleavedAddrGen<is_dram> addr_gen;
    addr_gen.bank_base_address = base_addr;
    addr_gen.page_size = page_size;
    uint64_t dst_addr_offset = 0;  // Offset into page.

    DPRINT << "process_write_paged - pages: " << pages << " page_size: " << page_size
           << " dispatch_cb_page_size: " << dispatch_cb_page_size;
    DPRINT << " start_page: " << page_id << " base_addr: " << HEX() << base_addr << DEC() << ENDL();

    while (write_length != 0) {
        // TODO #7360: Have more performant handling when page_size > dispatch_cb_page_size by not doing multiple writes
        // for one buffer page
        uint32_t xfer_size =
            page_size > dispatch_cb_page_size ? min(dispatch_cb_page_size, page_size - dst_addr_offset) : page_size;
        uint64_t dst = addr_gen.get_noc_addr(
            page_id, dst_addr_offset);  // XXXX replace this w/ walking the banks to save mul on GS

        // Get a Dispatch page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    uint32_t orphan_size = dispatch_cb_end - data_ptr;
                    if (orphan_size != 0) {
                        noc_async_write<dispatch_cb_page_size>(data_ptr, dst, orphan_size);
                        block_noc_writes_to_clear[rd_block_idx]++;
                        write_length -= orphan_size;
                        xfer_size -= orphan_size;
                        dst_addr_offset += orphan_size;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                    dst = addr_gen.get_noc_addr(page_id, dst_addr_offset);
                }
                move_rd_to_next_block<dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_noc_xy, my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx);
            cb_fence += n_pages * dispatch_cb_page_size;

            // Release pages for prefetcher
            // Since we gate how much we acquire to < 1/4 the buffer, this should be called enough
            cb_block_release_pages<
                upstream_noc_xy,
                upstream_dispatch_cb_sem_id,
                dispatch_cb_blocks,
                dispatch_cb_pages_per_block>(block_noc_writes_to_clear, wr_block_idx);
        }

        noc_async_write<dispatch_cb_page_size>(data_ptr, dst, xfer_size);
        block_noc_writes_to_clear[rd_block_idx]++;  // XXXXX maybe just write the noc internal api counter

        // If paged write is not completed for a page (dispatch_cb_page_size < page_size) then add offset, otherwise
        // incr page_id.
        if (dst_addr_offset + xfer_size < page_size) {
            dst_addr_offset += xfer_size;
        } else {
            page_id++;
            dst_addr_offset = 0;
        }

        write_length -= xfer_size;
        data_ptr += xfer_size;
    }

    cmd_ptr = data_ptr;
}

// Packed write command
// Layout looks like:
//   - CQDispatchCmd struct
//   - count CQDispatchWritePackedSubCmd structs (max 1020)
//   - pad to L1 alignment
//   - count data packets of size size, each L1 aligned
//
// Note that there are multiple size restrictions on this cmd:
//  - all sub_cmds fit in one page
//  - size fits in one page
//
// Since all subcmds all appear in the first page and given the size restrictions
// this command can't be too many pages.  All pages are released at the end
template <bool mcast, typename WritePackedSubCmd>
void process_write_packed(uint32_t flags) {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t count = cmd->write_packed.count;
    ASSERT(count <= (mcast ? max_write_packed_cores / 2 : max_write_packed_cores));
    constexpr uint32_t sub_cmd_size = sizeof(WritePackedSubCmd);
    // Copying in a burst is about a 30% net gain vs reading one value per loop below
    careful_copy_from_l1_to_local_cache(
        (volatile uint32_t tt_l1_ptr *)(cmd_ptr + sizeof(CQDispatchCmd)), count * sub_cmd_size / sizeof(uint32_t));

    uint32_t xfer_size = cmd->write_packed.size;
    uint32_t dst_addr = cmd->write_packed.addr;

    ASSERT(xfer_size <= dispatch_cb_page_size);

    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd) + count * sizeof(WritePackedSubCmd);
    data_ptr = round_up_pow2(data_ptr, L1_NOC_ALIGNMENT);
    uint32_t stride =
        (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE) ? 0 : round_up_pow2(xfer_size, L1_NOC_ALIGNMENT);
    DPRINT << data_ptr << " " << cmd_ptr << " " << xfer_size << " " << dispatch_cb_page_size << ENDL();
    ASSERT(stride != 0 || data_ptr - cmd_ptr + xfer_size <= dispatch_cb_page_size);

    volatile uint32_t tt_l1_ptr *l1_addr = (uint32_t *)(cmd_ptr + sizeof(CQDispatchCmd));
    cq_noc_async_write_init_state<CQ_NOC_snDL, mcast>(0, dst_addr, xfer_size);

    DPRINT << "dispatch_write_packed: " << xfer_size << " " << stride << " " << data_ptr << " " << count << ENDL();
    uint32_t writes = 0;
    uint32_t mcasts = 0;
    WritePackedSubCmd *sub_cmd_ptr = (WritePackedSubCmd *)l1_cache;
    while (count != 0) {
        uint32_t dst_noc = sub_cmd_ptr->noc_xy_addr;
        uint32_t num_dests = mcast ? ((CQDispatchWritePackedMulticastSubCmd *)sub_cmd_ptr)->num_mcast_dests : 1;
        sub_cmd_ptr++;
        uint64_t dst = get_noc_addr_helper(dst_noc, dst_addr);
        // Get a page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion
            uint32_t orphan_size = 0;
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    ASSERT(cb_fence == dispatch_cb_end);
                    orphan_size = cb_fence - data_ptr;
                    if (orphan_size != 0) {
                        cq_noc_async_write_with_state<CQ_NOC_SNdL>(data_ptr, dst, orphan_size, num_dests);
                        writes++;
                        mcasts += num_dests;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }

                block_noc_writes_to_clear[rd_block_idx] += writes;
                noc_nonposted_writes_num_issued[noc_index] += writes;
                noc_nonposted_writes_acked[noc_index] += mcasts;
                writes = 0;
                mcasts = 0;
                move_rd_to_next_block<dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_noc_xy, my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx);
            cb_fence += n_pages * dispatch_cb_page_size;

            // This is done here so the common case doesn't have to restore the pointers
            if (orphan_size != 0) {
                uint32_t remainder_xfer_size = xfer_size - orphan_size;
                uint32_t remainder_dst_addr = dst_addr + orphan_size;
                uint64_t remainder_dst = get_noc_addr_helper(dst_noc, remainder_dst_addr);
                cq_noc_async_write_with_state<CQ_NOC_SnDL>(data_ptr, remainder_dst, remainder_xfer_size, num_dests);
                // Reset values expected below
                cq_noc_async_write_with_state<CQ_NOC_snDL, CQ_NOC_WAIT, CQ_NOC_send>(0, dst, xfer_size);
                writes++;
                mcasts += num_dests;

                count--;
                data_ptr += stride - orphan_size;

                continue;
            }
        }

        cq_noc_async_write_with_state<CQ_NOC_SNdl>(data_ptr, dst, xfer_size, num_dests);
        writes++;
        mcasts += num_dests;

        count--;
        data_ptr += stride;
    }

    block_noc_writes_to_clear[rd_block_idx] += writes;
    noc_nonposted_writes_num_issued[noc_index] += writes;
    noc_nonposted_writes_acked[noc_index] += mcasts;
    // Release pages for prefetcher
    // write_packed releases pages at the end so the first page (w/ the sub_cmds) remains valid
    cb_block_release_pages<
        upstream_noc_xy,
        upstream_dispatch_cb_sem_id,
        dispatch_cb_blocks,
        dispatch_cb_pages_per_block>(block_noc_writes_to_clear, wr_block_idx);

    cmd_ptr = data_ptr;
}

static uint32_t process_debug_cmd(uint32_t cmd_ptr) {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t checksum = 0;
    uint32_t *data = (uint32_t *)((uint32_t)cmd + (uint32_t)sizeof(CQDispatchCmd));
    uint32_t size = cmd->debug.size;
    DPRINT << "checksum: " << cmd->debug.size << ENDL();

    // Dispatch checksum only handles running checksum on a single page
    // Host code prevents larger from flowing through
    // This way this code doesn't have to fetch multiple pages and then run
    // a cmd within those pages (messing up the implementation of that command)
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        checksum += *data++;
    }

    if (checksum != cmd->debug.checksum) {
        DEBUG_STATUS("!CHK");
        ASSERT(0);
    }

    return cmd_ptr + cmd->debug.stride;
}

static void process_wait() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t barrier = cmd->wait.barrier;
    uint32_t notify_prefetch = cmd->wait.notify_prefetch;
    uint32_t clear_count = cmd->wait.clear_count;
    uint32_t wait = cmd->wait.wait;
    uint32_t addr = cmd->wait.addr;
    uint32_t count = cmd->wait.count;

    if (barrier) {
        noc_async_write_barrier();
    }

    DEBUG_STATUS("PWW");
    volatile tt_l1_ptr uint32_t *sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(addr);
    DPRINT << " DISPATCH WAIT " << HEX() << addr << DEC() << " count " << count << ENDL();
#if defined(COMPILE_FOR_IDLE_ERISC)
    uint32_t heartbeat = 0;
#endif
    if (wait) {
        while (!wrap_ge(*sem_addr, count)) {
#if defined(COMPILE_FOR_IDLE_ERISC)
            RISC_POST_HEARTBEAT(heartbeat);
#endif
        }
    }
    DEBUG_STATUS("PWD");

    if (clear_count) {
        uint32_t neg_sem_val = -(*sem_addr);
        noc_semaphore_inc(get_noc_addr_helper(my_noc_xy, addr), neg_sem_val);
        noc_async_atomic_barrier();
    }

    if (notify_prefetch) {
        noc_semaphore_inc(get_noc_addr_helper(upstream_noc_xy, get_semaphore(upstream_sync_sem)), 1);
    }

    cmd_ptr += sizeof(CQDispatchCmd);
}

static void process_delay_cmd() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t count = cmd->delay.delay;
    for (volatile uint32_t i = 0; i < count; i++);
    cmd_ptr += sizeof(CQDispatchCmd);
}

static inline bool process_cmd_d(uint32_t &cmd_ptr) {
    bool done = false;

re_run_command:
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;

    switch (cmd->base.cmd_id) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR:
            DEBUG_STATUS("DWB");
            DPRINT << "cmd_write\n";
            process_write();
            DEBUG_STATUS("DWD");
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H:
            DPRINT << "cmd_write_linear_h\n";
            if (is_h_variant) {
                process_write();
            } else {
                relay_write_h();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
            DPRINT << "cmd_write_linear_h_host\n";
            if (is_h_variant) {
                process_write_host_h();
            } else {
                process_write_host_d();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PAGED:
            DPRINT << "cmd_write_paged is_dram: " << (uint32_t)cmd->write_paged.is_dram << ENDL();
            if (cmd->write_paged.is_dram) {
                process_write_paged<true>();
            } else {
                process_write_paged<false>();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PACKED: {
            DPRINT << "cmd_write_packed" << ENDL();
            uint32_t flags = cmd->write_packed.flags;
            if (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST) {
                process_write_packed<true, CQDispatchWritePackedMulticastSubCmd>(flags);
            } else {
                process_write_packed<false, CQDispatchWritePackedUnicastSubCmd>(flags);
            }
        } break;

        case CQ_DISPATCH_CMD_WAIT:
            DPRINT << "cmd_wait" << ENDL();
            process_wait();
            break;

        case CQ_DISPATCH_CMD_GO: DPRINT << "cmd_go" << ENDL(); break;

        case CQ_DISPATCH_CMD_SINK: DPRINT << "cmd_sink" << ENDL(); break;

        case CQ_DISPATCH_CMD_DEBUG:
            DPRINT << "cmd_debug" << ENDL();
            cmd_ptr = process_debug_cmd(cmd_ptr);
            goto re_run_command;
            break;

        case CQ_DISPATCH_CMD_DELAY:
            DPRINT << "cmd_delay" << ENDL();
            process_delay_cmd();
            break;

        case CQ_DISPATCH_CMD_TERMINATE:
            DPRINT << "dispatch terminate\n";
            if (is_d_variant && !is_h_variant) {
                relay_to_next_cb<split_dispatch_page_preamble_size>(cmd_ptr, sizeof(CQDispatchCmd));
            }
            cmd_ptr += sizeof(CQDispatchCmd);
            done = true;
            break;

        default:
            DPRINT << "dispatcher_d invalid command:" << cmd_ptr << " " << cb_fence << " " << dispatch_cb_base << " "
                   << dispatch_cb_end << " " << rd_block_idx << " "
                   << "xx" << ENDL();
            DPRINT << HEX() << *(uint32_t *)cmd_ptr << ENDL();
            DPRINT << HEX() << *((uint32_t *)cmd_ptr + 1) << ENDL();
            DPRINT << HEX() << *((uint32_t *)cmd_ptr + 2) << ENDL();
            DPRINT << HEX() << *((uint32_t *)cmd_ptr + 3) << ENDL();
            DEBUG_STATUS("!CMD");
            ASSERT(0);
    }

    return done;
}

static inline bool process_cmd_h(uint32_t &cmd_ptr) {
    bool done = false;

    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;

    switch (cmd->base.cmd_id) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR_H:
            DPRINT << "dispatch_h write_linear_h\n";
            process_write();
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
            DPRINT << "dispatch_h linear_h_host\n";
            process_write_host_h();
            break;

        case CQ_DISPATCH_CMD_TERMINATE:
            DPRINT << "dispatch_h terminate\n";
            cmd_ptr += sizeof(CQDispatchCmd);
            done = true;
            break;

        default:
            DPRINT << "dispatcher_h invalid command:" << cmd_ptr << " " << cb_fence << " "
                   << " " << dispatch_cb_base << " " << dispatch_cb_end << " " << rd_block_idx << " "
                   << "xx" << ENDL();
            DPRINT << HEX() << *(uint32_t *)cmd_ptr << ENDL();
            DPRINT << HEX() << *((uint32_t *)cmd_ptr + 1) << ENDL();
            DPRINT << HEX() << *((uint32_t *)cmd_ptr + 2) << ENDL();
            DPRINT << HEX() << *((uint32_t *)cmd_ptr + 3) << ENDL();
            DEBUG_STATUS("!CMD");
            ASSERT(0);
    }

    return done;
}

void kernel_main() {
    DPRINT << "dispatch_" << is_h_variant << is_d_variant << ": start" << ENDL();

    static_assert(is_d_variant || split_dispatch_page_preamble_size == 0);

    for (uint32_t i = 0; i < dispatch_cb_blocks; i++) {
        uint32_t next_block = i + 1;
        uint32_t offset = next_block * dispatch_cb_pages_per_block * dispatch_cb_page_size;
        block_next_start_addr[i] = dispatch_cb_base + offset;
    }

    cb_fence = dispatch_cb_base;
    rd_block_idx = 0;
    wr_block_idx = 0;
    block_noc_writes_to_clear[0] = noc_nonposted_writes_num_issued[noc_index] + 1;
    cmd_ptr = dispatch_cb_base;

    {
        uint32_t completion_queue_wr_ptr_and_toggle = *get_cq_completion_write_ptr();
        cq_write_interface.completion_fifo_wr_ptr = completion_queue_wr_ptr_and_toggle & 0x7fffffff;
        cq_write_interface.completion_fifo_wr_toggle = completion_queue_wr_ptr_and_toggle >> 31;
    }
    bool done = false;
    while (!done) {
        DeviceZoneScopedND("CQ-DISPATCH", block_noc_writes_to_clear, rd_block_idx );
        if (cmd_ptr == cb_fence) {
            get_cb_page<
                dispatch_cb_base,
                dispatch_cb_blocks,
                dispatch_cb_log_page_size,
                my_noc_xy,
                my_dispatch_cb_sem_id>(
                cmd_ptr, cb_fence, block_noc_writes_to_clear, block_next_start_addr, rd_block_idx);
        }

        done = is_d_variant ? process_cmd_d(cmd_ptr) : process_cmd_h(cmd_ptr);

        // Move to next page
        cmd_ptr = round_up_pow2(cmd_ptr, dispatch_cb_page_size);

        // XXXXX move this inside while loop waiting for get_dispatch_cb_page above
        // XXXXX can potentially clear a partial block when stalled w/ some more bookkeeping
        cb_block_release_pages<
            upstream_noc_xy,
            upstream_dispatch_cb_sem_id,
            dispatch_cb_blocks,
            dispatch_cb_pages_per_block>(block_noc_writes_to_clear, wr_block_idx);
    }

    noc_async_write_barrier();

    if (is_h_variant && !is_d_variant) {
        // Set upstream semaphore MSB to signal completion and path teardown
        // in case dispatch_h is connected to a depacketizing stage.
        // TODO: This should be replaced with a signal similar to what packetized
        // components use.
        noc_semaphore_inc(get_noc_addr_helper(upstream_noc_xy, get_semaphore(upstream_dispatch_cb_sem_id)), 0x80000000);
    }

#if defined(COMPILE_FOR_IDLE_ERISC)
    uint32_t heartbeat = 0;
    RISC_POST_HEARTBEAT(heartbeat);
#endif

    // Release any held pages from the last block
    if (rd_block_idx != wr_block_idx) {
        // We're 1 block behind
        cb_release_pages<upstream_noc_xy, upstream_dispatch_cb_sem_id>(dispatch_cb_pages_per_block);
    }
    uint32_t npages =
        dispatch_cb_pages_per_block - ((block_next_start_addr[rd_block_idx] - cmd_ptr) >> dispatch_cb_log_page_size);
    cb_release_pages<upstream_noc_xy, upstream_dispatch_cb_sem_id>(npages);

    // Confirm expected number of pages, spinning here is a leak
    cb_wait_all_pages<my_dispatch_cb_sem_id>(0);

    DPRINT << "dispatch_" << is_h_variant << is_d_variant << ": out" << ENDL();
}
