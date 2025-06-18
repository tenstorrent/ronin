// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "common/tt_backend_api_types.hpp"
#include "common/utils.hpp"
#include "common/core_coord.h"
#include "jit_build/data_format.hpp"
#include "jit_build/settings.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tt_stl/aligned_allocator.hpp"
#include "llrt/rtoptions.hpp"


namespace tt::tt_metal {

static constexpr uint32_t CACHE_LINE_ALIGNMENT = 64;

template <typename T>
using vector_cache_aligned = std::vector<T, tt::stl::aligned_allocator<T, CACHE_LINE_ALIGNMENT>>;

class JitBuildSettings;

enum class JitBuildProcessorType {
    DATA_MOVEMENT,
    COMPUTE,
    ETHERNET
};

// The build environment
// Includes the path to the src/output and global defines, flags, etc
// Device specific
class JitBuildEnv {
    friend class JitBuildState;
    friend class JitBuildDataMovement;
    friend class JitBuildCompute;
    friend class JitBuildEthernet;

  public:
    JitBuildEnv();
    void init(uint32_t build_key, tt::ARCH arch);

    tt::ARCH get_arch() const { return arch_; }
    const string& get_root_path() const { return root_; }
    const string& get_out_root_path() const { return out_root_; }
    const string& get_out_firmware_root_path() const { return out_firmware_root_; }
    const string& get_out_kernel_root_path() const { return out_kernel_root_; }

  private:
    tt::ARCH arch_;
    string arch_name_;
    string aliased_arch_name_;

    // Paths
    string root_;
    string out_root_;
    string out_firmware_root_;
    string out_kernel_root_;

    // Tools
    string gpp_;
    string objcopy_;

    // Compilation options
    string cflags_;
    string defines_;
    string includes_;
    string lflags_;
};

// All the state used for a build in an abstract base class
// Contains everything needed to do a build (all settings, methods, etc)
class JitBuildState {
public:
    enum class TargetId {
        BRISC,
        NCRISC,
        TRISC0,
        TRISC1,
        TRISC2,
        ERISC,
        IDLE_ERISC
    };

protected:
    const JitBuildEnv &env_;

    int core_id_;
    int is_fw_;

    std::string out_path_;
    TargetId target_id_;
    std::string target_name_;
    std::string target_full_path_;

    std::string defines_;

public:
    JitBuildState(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw = false);
    virtual ~JitBuildState() = default;

    void finish_init();

    void build(
        const JitBuildSettings *settings, 
        const std::string &kernel_in_path) const;

    std::string get_out_path() const { 
        return this->out_path_; 
    };
    TargetId get_target_id() const {
        return this->target_id_;
    }
    std::string get_target_name() const { 
        return this->target_name_; 
    };
    std::string get_target_out_path(const string &kernel_name) const { 
        return this->out_path_ + kernel_name + target_full_path_; 
    }

private:
    void build_kernel(
        const JitBuildSettings *settings, 
        const std::string &kernel_in_path, 
        const std::string &target_out_path,
        const std::string &temp_dir) const;
    std::string make_compiler_defines(const JitBuildSettings *settings) const;
};

// Set of build states
// Used for parallel builds, builds all members in one call
typedef vector<std::shared_ptr<JitBuildState>> JitBuildStateSet;

// Exracts a slice of builds from a JitBuildState
// Used for parallel building a subset of the builds in a JitBuildStateSet
struct JitBuildStateSubset {
    const std::shared_ptr<JitBuildState> * build_ptr;
    int size;
};

// Specific build types
// These specialize a JitBuildState with everything need to build for a target
class JitBuildDataMovement : public JitBuildState {
  private:

  public:
    JitBuildDataMovement(const JitBuildEnv& env, int which, bool is_fw = false);
};

class JitBuildCompute : public JitBuildState {
  private:
  public:
    JitBuildCompute(const JitBuildEnv& env, int which, bool is_fw = false);
};

class JitBuildEthernet : public JitBuildState {
  private:
  public:
    JitBuildEthernet(const JitBuildEnv& env, int which, bool is_fw = false);
};

// Abstract base class for kernel specialization
// Higher levels of the SW derive from this and fill in build details not known to the build system
// (eg, API specified settings)
class JitBuildSettings {
  public:
    virtual const string& get_full_kernel_name() const = 0;
    virtual void process_defines(const std::function<void (const string& define, const string &value)>) const = 0;
    virtual void process_compile_time_args(const std::function<void (int i, uint32_t value)>) const = 0;
  private:
    bool use_multi_threaded_compile = true;
};

void jit_build(const JitBuildState& build, const JitBuildSettings *settings, const string& kernel_in_path);
void jit_build_set(const JitBuildStateSet& builds, const JitBuildSettings *settings, const string& kernel_in_path);
void jit_build_subset(const JitBuildStateSubset& builds, const JitBuildSettings *settings, const string& kernel_in_path);

inline void launch_build_step(const std::function<void()> build_func) {
    build_func();
}

} // namespace tt::tt_metal
