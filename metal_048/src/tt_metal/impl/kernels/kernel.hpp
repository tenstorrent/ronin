// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>
#include <optional>
#include <variant>
#include <type_traits>
#include <memory>

#include "jit_build/build.hpp"
#include "common/base_types.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/llrt/tt_memory.h"

namespace tt {

namespace tt_metal {

using Config = std::variant<DataMovementConfig, EthernetConfig, ComputeConfig>;

struct RuntimeArgsData {
    uint32_t * rt_args_data;
    size_t rt_args_size;

    inline uint32_t & operator[](size_t index) {
        TT_ASSERT(index < rt_args_size, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline const uint32_t& operator[](size_t index) const {
        TT_ASSERT(index < rt_args_size, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline uint32_t & at(size_t index) {
        TT_FATAL(index < rt_args_size, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline const uint32_t& at(size_t index) const {
        TT_FATAL(index < rt_args_size, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline uint32_t * data() noexcept {
        return rt_args_data;
    }
    inline const uint32_t * data() const noexcept {
        return rt_args_data;
    }
    inline size_t size() const noexcept{
        return rt_args_size;
    }
};

class Kernel : public JitBuildSettings {
   public:
    Kernel(const std::string &kernel_path_file_name, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &compile_args, const std::map<std::string, std::string>&defines);

    virtual ~Kernel() {}

    std::string kernel_path_file_name() const { return kernel_path_file_name_; }

    std::string name() const;

    CoreRangeSet core_range_set() const { return core_range_set_; }

    const std::set<CoreCoord> &logical_cores() const;

    std::vector<CoreRange> logical_coreranges() const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    std::vector<ll_api::memory> const &binaries(uint32_t build_key) const;

    std::vector<uint32_t> compile_time_args() const { return compile_time_args_; }

    const std::set<CoreCoord>& cores_with_runtime_args() const { return core_with_runtime_args_; }

    std::vector<uint32_t> & runtime_args(const CoreCoord &logical_core);
    RuntimeArgsData & runtime_args_data(const CoreCoord &logical_core);
    std::vector< std::vector< std::vector<uint32_t>> > & runtime_args();
    std::vector< std::vector< RuntimeArgsData > > & runtime_args_data();
    std::vector<uint32_t> & common_runtime_args();
    RuntimeArgsData & common_runtime_args_data();

    std::map<std::string, std::string> defines() const { return defines_; }

    virtual RISCV processor() const = 0;

    virtual bool configure(Device *device, const CoreCoord &logical_core) const = 0;

    virtual Config config() const = 0;

    std::string compute_hash() const;
    virtual void set_build_options(JitBuildOptions &build_options) const = 0;
    virtual void generate_binaries(Device *device, JitBuildOptions& build_options) const = 0;
    inline uint16_t get_binary_size16() const { return binary_size16_; }
    void set_binary_path ( const std::string & binary_path) { binary_path_ = binary_path; }
    void set_binaries(uint32_t build_key, std::vector<ll_api::memory> &&binaries);
    uint32_t get_common_runtime_args_offset();
    void set_common_runtime_args_offset();
    virtual void read_binaries(Device *device) = 0;

    void validate_runtime_args_size(size_t num_unique_rt_args, size_t num_common_rt_args, const CoreCoord& logical_core);
    void set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);
    void set_common_runtime_args(const std::vector<uint32_t> &runtime_args);

    int get_watcher_kernel_id() { return watcher_kernel_id_; }

    CoreType get_kernel_core_type() const;
    void set_full_name(const string& s) { kernel_full_name_ = s; }
    const string& get_full_kernel_name() const override;
    void process_defines(const std::function<void (const string& define, const string &value)>) const override;
    void process_compile_time_args(const std::function<void (int i, uint32_t value)>) const override;

   protected:
    const int watcher_kernel_id_;
    std::string kernel_path_file_name_;                 // Full kernel path and file name
    std::string kernel_full_name_;                      // Name + hash
    CoreRangeSet core_range_set_;
    std::string binary_path_;
    // DataMovement kernels have one binary each and Compute kernels have three binaries
    // Different set of binaries per device because kernel compilation is device dependent
    // TODO: break this dependency by https://github.com/tenstorrent/tt-metal/issues/3381
    std::unordered_map<chip_id_t, std::vector<ll_api::memory>> binaries_;
    uint16_t binary_size16_;
    std::vector<uint32_t> compile_time_args_;
    std::vector< std::vector< std::vector<uint32_t>> > core_to_runtime_args_;
    std::vector< std::vector< RuntimeArgsData> > core_to_runtime_args_data_;
    std::vector<uint32_t> common_runtime_args_;
    RuntimeArgsData common_runtime_args_data_;
    std::set<CoreCoord> core_with_runtime_args_;
    std::size_t max_runtime_args_per_core_;             // For validation
    CoreCoord core_with_max_runtime_args_;              // For validation
    std::map<std::string, std::string> defines_;        // preprocessor defines. this is to be able to generate generic instances.
    std::set<CoreCoord> logical_cores_;

    virtual uint8_t expected_num_binaries() const = 0;

    virtual std::string config_hash() const = 0;

    virtual std::pair<uint64_t, uint64_t> get_runtime_args_range() const = 0;
};

class DataMovementKernel : public Kernel {
   public:
    DataMovementKernel(const std::string &kernel_path, const CoreRangeSet &cr_set, const DataMovementConfig &config) : Kernel(kernel_path, cr_set, config.compile_args, config.defines), config_(config) {}

    ~DataMovementKernel() {}

    RISCV processor() const override;

    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(Device *device, JitBuildOptions& build_options) const override;
    void read_binaries(Device *device) override;

    bool configure(Device *device, const CoreCoord &logical_core) const override;

    Config config() const override { return this->config_; }

    void process_defines(const std::function<void (const string& define, const string &value)>) const override;

   private:
    const DataMovementConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;

    std::pair<uint64_t, uint64_t> get_runtime_args_range() const override;
};

class EthernetKernel : public Kernel {
   public:
    EthernetKernel(const std::string &kernel_path, const CoreRangeSet &cr_set, const EthernetConfig &config) :
        Kernel(kernel_path, cr_set, config.compile_args, config.defines), config_(config) {}

    ~EthernetKernel() {}

    RISCV processor() const override;

    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(Device *device, JitBuildOptions& build_options) const override;
    void read_binaries(Device *device) override;

    bool configure(Device *device, const CoreCoord &logical_core) const override;

    Config config() const override { return this->config_; }

    void process_defines(const std::function<void (const string& define, const string &value)>) const override;

   private:
    const EthernetConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;

    std::pair<uint64_t, uint64_t> get_runtime_args_range() const override;
};

class ComputeKernel : public Kernel {
   public:
    ComputeKernel(const std::string &kernel_path, const CoreRangeSet &cr_set, const ComputeConfig &config) : Kernel(kernel_path, cr_set, config.compile_args, config.defines), config_(config) {}

    ~ComputeKernel() {}

    RISCV processor() const override;

    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(Device *device, JitBuildOptions& build_options) const override;
    void read_binaries(Device *device) override;

    bool configure(Device *device, const CoreCoord &logical_core) const override;

    Config config() const override { return this->config_; }

    void process_defines(const std::function<void (const string& define, const string &value)>) const override;

   private:
    const ComputeConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;

    std::pair<uint64_t, uint64_t> get_runtime_args_range() const override;
};

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor);

struct KernelDefinesHash {
    KernelDefinesHash() {}

    size_t operator()(const std::map<std::string, std::string> &c_defines) const;
};

}  // namespace tt_metal

}  // namespace tt
