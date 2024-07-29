// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
//#include <mutex> // [RONIN]
#include <optional>

#include "common/bfloat16.hpp"
#include "common/core_coord.h"
#include "common/tt_backend_api_types.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h" // For CoreType
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

class Device;

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY,
    L1_SMALL,
};

enum class TensorMemoryLayout {
    INTERLEAVED,
    SINGLE_BANK,
    HEIGHT_SHARDED,
    WIDTH_SHARDED,
    BLOCK_SHARDED,
};

enum class ShardOrientation {
    ROW_MAJOR,
    COL_MAJOR,
};

struct ShardSpec {
    CoreRangeSet grid;
    std::array<uint32_t, 2> shape;
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    bool halo = false;

    ShardSpec(
        const CoreRangeSet &core_sets_,
        const std::array<uint32_t, 2> &shard_shape_,
        const ShardOrientation &shard_orientation_ = ShardOrientation::ROW_MAJOR,
        const bool &halo_ = false) :
        grid(core_sets_), shape(shard_shape_), orientation(shard_orientation_), halo(halo_) {
        ;
    }

    const uint32_t num_cores() const { return this->grid.num_cores(); }
    const uint32_t numel() const { return this->shape[0] * this->shape[1]; }

    static constexpr auto attribute_names = std::forward_as_tuple("grid", "shape", "orientation", "halo");
    constexpr auto attribute_values() const {
        return std::forward_as_tuple(this->grid, this->shape, this->orientation, this->halo);
    }
};

bool operator==(const ShardSpec &spec_a, const ShardSpec &spec_b);
bool operator!=(const ShardSpec &spec_a, const ShardSpec &spec_b);

struct ShardSpecBuffer {
    ShardSpec tensor_shard_spec;
    std::array<uint32_t, 2> page_shape;
    std::array<uint32_t, 2> tensor2d_shape;
    ShardSpecBuffer(
        const CoreRangeSet &core_sets_,
        const std::array<uint32_t, 2> &shard_shape_,
        const ShardOrientation &shard_orientation_,
        const bool &halo_,
        const std::array<uint32_t, 2> &page_shape,
        const std::array<uint32_t, 2> &tensor2d_shape) :
        tensor_shard_spec(core_sets_, shard_shape_, shard_orientation_, halo_) {
        this->page_shape = page_shape;
        this->tensor2d_shape = tensor2d_shape;
    }
    ShardSpecBuffer(
        const ShardSpec &shard_spec,
        const std::array<uint32_t, 2> &page_shape,
        const std::array<uint32_t, 2> &tensor2d_shape) :
        tensor_shard_spec(shard_spec) {
        this->page_shape = page_shape;
        this->tensor2d_shape = tensor2d_shape;
    }
    CoreRangeSet grid() const { return tensor_shard_spec.grid; }
    std::array<uint32_t, 2> shape() const { return tensor_shard_spec.shape; }
    ShardOrientation orientation() const { return tensor_shard_spec.orientation; }
    bool halo() const { return tensor_shard_spec.halo; }
    std::array<uint32_t, 2> shape_in_pages() const {
        auto width_in_pages = tensor_shard_spec.shape[0] / page_shape[0];
        auto height_in_pages = tensor_shard_spec.shape[1] / page_shape[1];
        return {width_in_pages, height_in_pages};
    }
    uint32_t size() const {
        auto shape_in_pages_ = this->shape_in_pages();
        return shape_in_pages_[0] * shape_in_pages_[1];
    }
};

struct BufferConfig {
    Device *device;
    uint64_t size;       // Size in bytes
    uint64_t page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED;
};

typedef BufferConfig InterleavedBufferConfig;

// copied from above instead of using inheritance such that we can use
// designator constructor
struct ShardedBufferConfig {
    Device *device;
    uint64_t size;       // Size in bytes
    uint64_t page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type = BufferType::L1;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardSpecBuffer shard_parameters;
};

bool is_sharded(const TensorMemoryLayout &layout);

struct BufferPageMapping {
    std::vector<CoreCoord> all_cores_;
    std::vector<uint32_t> core_bank_indices_;
    std::vector<std::vector<uint32_t>> core_host_page_indices_;
    std::vector<uint32_t> dev_page_to_core_mapping_;

    // some dev pages don't have mapping to host (in case of padding)
    std::vector<std::optional<uint32_t>> dev_page_to_host_page_mapping_;
    std::vector<uint32_t> host_page_to_dev_page_mapping_;
    std::unordered_map<CoreCoord, uint32_t> core_to_core_id_;
    std::vector<uint32_t> host_page_to_local_shard_page_mapping_;
    std::vector<std::array<uint32_t, 2>> core_shard_shape_;
};

class Buffer {
   public:
    Buffer() :
        device_(nullptr),
        buffer_type_(BufferType::DRAM),
        buffer_layout_(TensorMemoryLayout::INTERLEAVED),
        shard_parameters_(std::nullopt) {}

    Buffer(
        Device *device,
        uint64_t size,
        uint64_t page_size,
        const BufferType buffer_type,
        const TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
        const std::optional<ShardSpecBuffer>& shard_parameter = std::nullopt,
        bool allocate = true);

    Buffer(const Buffer &other);
    Buffer &operator=(const Buffer &other);

    Buffer(Buffer &&other);
    Buffer &operator=(Buffer &&other);

    ~Buffer();
    Device *device() const { return device_; }

    uint32_t size() const { return static_cast<uint32_t>(size_); }

    void set_size(uint64_t size) { size_ = size; }
    // Returns address of buffer in the first bank
    uint32_t address() const { return static_cast<uint32_t>(address_); }

    void set_address(uint64_t addr) { address_ = addr; }

    uint32_t page_size() const { return page_size_; }

    uint32_t num_pages() const { return this->size() / this->page_size(); }

    uint32_t num_dev_pages() const {
        if (!is_sharded(this->buffer_layout_)) {
            return this->num_pages();
        } else {
            return this->shard_spec().size() * this->num_cores();
        }
    }

    BufferType buffer_type() const { return buffer_type_; }
    CoreType core_type() const {
        switch (this->buffer_type_) {
            case BufferType::DRAM:
                return CoreType::DRAM;
            case BufferType::L1:
            case BufferType::L1_SMALL:
                return CoreType::WORKER;
            default:
                TT_THROW("Unknown CoreType for buffer");
        }
    }

    bool is_l1() const { return buffer_type() == BufferType::L1 or buffer_type() == BufferType::L1_SMALL; }
    bool is_dram() const { return buffer_type() == BufferType::DRAM; }

    TensorMemoryLayout buffer_layout() const { return buffer_layout_; }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    CoreCoord noc_coordinates(uint32_t bank_id) const;

    // returns NoC coordinates of first bank buffer is in
    CoreCoord noc_coordinates() const;

    uint64_t page_address(uint32_t bank_id, uint32_t page_index) const;

    // SHARDED API STARTS HERE
    // TODO: WILL SEPARATE INTO SHARDED BUFFER CLASS

    uint64_t sharded_page_address(uint32_t bank_id, uint32_t page_index) const;

    ShardSpecBuffer shard_spec() const {
        TT_ASSERT(is_sharded(this->buffer_layout_), "Buffer not sharded");
        TT_ASSERT(shard_parameters_.has_value());
        return this->shard_parameters_.value();
    }

    uint32_t num_cores() const {
        if (!is_sharded(this->buffer_layout_))
            return 1;
        else {
            return this->shard_spec().tensor_shard_spec.grid.num_cores();
        }
    }

   private:
    virtual void allocate();

    virtual void deallocate();
    friend void DeallocateBuffer(Buffer &buffer);

    uint64_t translate_page_address(uint64_t offset, uint32_t bank_id) const;

    Device *device_;
    uint64_t size_;       // Size in bytes
    uint64_t address_;    // Address of buffer
    uint64_t page_size_;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type_;
    TensorMemoryLayout buffer_layout_;
    std::optional<ShardSpecBuffer> shard_parameters_;
};

BufferPageMapping generate_buffer_page_mapping(const Buffer &buffer);

namespace detail {
using PageAddress = uint32_t;
using Deviceid = uint32_t;

class buffer_map_t {
   public:
    void insert(std::tuple<Deviceid, PageAddress> buf_attr, Buffer *buffer) {
#if 0 // [RONIN]
        std::scoped_lock<std::mutex> lock(this->map_mutex);
#endif
        this->map.insert({buf_attr, buffer});
    }

    void erase(std::tuple<Deviceid, PageAddress> buf_attr) {
#if 0 // [RONIN]
        std::scoped_lock<std::mutex> lock(this->map_mutex);
#endif
        this->map.erase(buf_attr);
    }

    std::map<std::tuple<Deviceid, PageAddress>, Buffer *> value() {
#if 0 // [RONIN]
        std::scoped_lock<std::mutex> lock(this->map_mutex);
#endif
        return this->map;
    }

    ~buffer_map_t() { TT_ASSERT(this->map.empty(), "Not all buffers deallocated by runtime!"); }

   private:
#if 0 // [RONIN]
    std::mutex map_mutex;
#endif
    std::map<std::tuple<Deviceid, PageAddress>, Buffer *> map = {};
};

extern buffer_map_t BUFFER_MAP;
}  // namespace detail

using HostDataType = std::variant<
    const std::shared_ptr<std::vector<uint16_t>>,
    const std::shared_ptr<std::vector<int32_t>>,
    const std::shared_ptr<std::vector<uint32_t>>,
    const std::shared_ptr<std::vector<float>>,
    const std::shared_ptr<std::vector<bfloat16>>,
    const void *>;

}  // namespace tt_metal

}  // namespace tt
