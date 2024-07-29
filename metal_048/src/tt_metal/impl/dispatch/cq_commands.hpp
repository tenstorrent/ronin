// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file is shared by host and device CQ dispatch

// Prefetcher/Dispatcher CMD interfaces
//  - CMD ID enums: identify the command to execute
//  - CMD structures: contain parameters for each command

#pragma once

#include <cstdint> // [RONIN]

// [RONIN]
#ifdef _MSC_VER
#define __attribute__(x)
#endif

constexpr uint32_t CQ_PREFETCH_CMD_BARE_MIN_SIZE = 32; // for NOC PCIe alignemnt
constexpr uint32_t CQ_DISPATCH_CMD_SIZE = 16;          // for L1 alignment

// Prefetcher CMD ID enums
enum CQPrefetchCmdId : uint8_t {
    CQ_PREFETCH_CMD_ILLEGAL = 0,              // common error value
    CQ_PREFETCH_CMD_RELAY_LINEAR = 1,         // relay banked/paged data from src_noc to dispatcher
    CQ_PREFETCH_CMD_RELAY_PAGED = 2,          // relay banked/paged data from src_noc to dispatcher
    CQ_PREFETCH_CMD_RELAY_INLINE = 3,         // relay (inline) data from CmdDatQ to dispatcher
    CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH = 4, // same as above, but doesn't flush the page to dispatcher
    CQ_PREFETCH_CMD_EXEC_BUF = 5,             // execute commands from a buffer
    CQ_PREFETCH_CMD_EXEC_BUF_END = 6,         // finish executing commands from a buffer (return)
    CQ_PREFETCH_CMD_STALL = 7,                // drain pipe through dispatcher
    CQ_PREFETCH_CMD_DEBUG = 8,                // log waypoint data to watcher, checksum
    CQ_PREFETCH_CMD_TERMINATE = 9             // quit
};

// Dispatcher CMD ID enums
enum CQDispatchCmdId : uint8_t {
    CQ_DISPATCH_CMD_ILLEGAL = 0,            // common error value
    CQ_DISPATCH_CMD_WRITE_LINEAR = 1,       // write data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WRITE_LINEAR_H = 2,     // write data from dispatcher to dst_noc on dispatch_h chip
    CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST = 3,// like write, dedicated to writing to host
    CQ_DISPATCH_CMD_WRITE_PAGED = 4,        // write banked/paged data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WRITE_PACKED = 5,       // write to multiple noc addresses with packed data
    CQ_DISPATCH_CMD_WAIT = 6,               // wait until workers are done
    CQ_DISPATCH_CMD_GO = 7,                 // send go message
    CQ_DISPATCH_CMD_SINK = 8,               // act as a data sink (for testing)
    CQ_DISPATCH_CMD_DEBUG = 9,              // log waypoint data to watcher, checksum
    CQ_DISPATCH_CMD_DELAY = 10,             // insert delay (for testing)
    CQ_DISPATCH_CMD_TERMINATE = 11          // quit
};

// RONIN
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif

//
//    Shared commands
//

struct CQGenericDebugCmd {
    uint8_t pad;
    uint16_t key;                          // prefetcher/dispatcher all write to watcher
    uint32_t checksum;                     // checksum of payload
    uint32_t size;                         // size of payload
    uint32_t stride;                       // stride to next Cmd (may be within the payload)
} __attribute__((packed));

//
//    Prefetcher CMD structures
//

struct CQPrefetchBaseCmd {
    enum CQPrefetchCmdId cmd_id;
} __attribute__((packed));

struct CQPrefetchRelayLinearCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t noc_xy_addr;
    uint32_t addr;
    uint32_t length;
} __attribute__((packed));;

constexpr uint32_t CQ_PREFETCH_RELAY_PAGED_START_PAGE_SHIFT = 0;
constexpr uint32_t CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT = 4;
constexpr uint32_t CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK = 0x0f;

struct CQPrefetchRelayPagedCmd {
    uint8_t packed_page_flags;  // start page and is_dram flag
    uint16_t length_adjust;     // bytes subtracted from size (multiple of 32)
    uint32_t base_addr;
    uint32_t page_size;
    uint32_t pages;
} __attribute__((packed));

struct CQPrefetchRelayInlineCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t length;
    uint32_t stride;          // explicit stride saves a few insns on device
} __attribute__((packed));

struct CQPrefetchExecBufCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t base_addr;
    uint32_t log_page_size;
    uint32_t pages;
} __attribute__((packed));

struct CQPrefetchCmd {
    CQPrefetchBaseCmd base;
    union {
        CQPrefetchRelayLinearCmd relay_linear;
        CQPrefetchRelayPagedCmd relay_paged;
        CQPrefetchRelayInlineCmd relay_inline;
        CQPrefetchExecBufCmd exec_buf;
        CQGenericDebugCmd debug;
    } __attribute__((packed));
};

//
//    Dispatcher CMD structures
//

struct CQDispatchBaseCmd {
    enum CQDispatchCmdId cmd_id;
} __attribute__((packed));

struct CQDispatchWriteCmd {
    uint8_t num_mcast_dests;    // 0 = unicast, 1+ = multicast
    uint16_t pad1;
    uint32_t noc_xy_addr;
    uint32_t addr;
    uint32_t length;
} __attribute__((packed));

struct CQDispatchWriteHostCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t pad3;
    uint32_t pad4;
    uint32_t length;
} __attribute__((packed));

struct CQDispatchWritePagedCmd {
    uint8_t is_dram;          // one flag, false=l1
    uint16_t start_page;
    uint32_t base_addr;
    uint32_t page_size;
    uint32_t pages;
} __attribute__((packed));


constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE      = 0x00;
constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST     = 0x01;
constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE = 0x02;

// 
//    count - (max 1020 unicast, 510 mcast).
//        Max num sub-cmds =
//            (dispatch_constants::TRANSFER_PAGE_SIZE - sizeof(CQDispatchCmd)) / 
//                sizeof(CQDispatchWritePacked * castSubCmd)
//    size - stride is padded to L1 alignment and less than dispatch_cb_page_size
//
struct CQDispatchWritePackedCmd {
    uint8_t flags;            // see above
    uint16_t count;           // number of sub-cmds (max 1020 unicast, 510 mcast). Max num sub-cmds
    uint32_t addr;            // common memory address across all packed SubCmds
    uint16_t size;            // size of each packet
} __attribute__((packed));

struct CQDispatchWritePackedUnicastSubCmd {
    uint32_t noc_xy_addr;     // unique XY address for each SubCmd
} __attribute__((packed));

struct CQDispatchWritePackedMulticastSubCmd {
    uint32_t noc_xy_addr;     // unique XY address for each SubCmd
    uint32_t num_mcast_dests;
} __attribute__((packed));

struct CQDispatchWaitCmd {
    uint8_t barrier;          // if true, issue write barrier
    uint8_t notify_prefetch;  // if true, inc prefetch sem
    uint8_t clear_count;      // if true, reset count to 0
    uint8_t wait;             // if true, wait on count value below
    uint8_t pad1;
    uint16_t pad2;
    uint32_t addr;            // address to read
    uint32_t count;           // wait while address is < count
} __attribute__((packed));

struct CQDispatchDelayCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t delay;
} __attribute__((packed));

struct CQDispatchCmd {
    CQDispatchBaseCmd base;

    union {
        CQDispatchWriteCmd write_linear;
        CQDispatchWriteHostCmd write_linear_host;
        CQDispatchWritePagedCmd write_paged;
        CQDispatchWritePackedCmd write_packed;
        CQDispatchWaitCmd wait;
        CQGenericDebugCmd debug;
        CQDispatchDelayCmd delay;
    } __attribute__((packed));
};

// RONIN
#ifdef _MSC_VER
#pragma pack(pop)
#endif

//
//    PrefetchH to PrefetchD packet header
//

struct CQPrefetchHToPrefetchDHeader {
    uint32_t length;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
    uint32_t pad4;
    uint32_t pad5;
    uint32_t pad6;
    uint32_t pad7;
};


// if this fails, padding above needs to be adjusted
static_assert(sizeof(CQPrefetchBaseCmd) == sizeof(uint8_t)); 
// if this fails, padding above needs to be adjusted
static_assert(sizeof(CQDispatchBaseCmd) == sizeof(uint8_t)); 
static_assert((sizeof(CQPrefetchCmd) & (CQ_DISPATCH_CMD_SIZE - 1)) == 0);
static_assert((sizeof(CQDispatchCmd) & (CQ_DISPATCH_CMD_SIZE - 1)) == 0);
static_assert((sizeof(CQPrefetchHToPrefetchDHeader) & (CQ_PREFETCH_CMD_BARE_MIN_SIZE - 1)) == 0);

// [RONIN]
#ifdef _MSC_VER
#undef __attribute__
#endif

