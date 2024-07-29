// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build/build.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
//#include <thread> // [RONIN]
#include <memory> // [RONIN]
#include <fstream> // [RONIN]

//#include "common/executor.hpp" // [RONIN]
#include "dev_mem_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "jit_build/genfiles.hpp"
#include "jit_build/kernel_args.hpp"
//#include "tools/profiler/common.hpp" // [RONIN]
//#include "tools/profiler/profiler_state.hpp" // [RONIN]
#include "tt_metal/impl/kernels/kernel.hpp"

#include "device/api/kernel_builder.hpp" // [RONIN]

using namespace std;
using namespace tt;

namespace tt::tt_metal {

#if 0 // [RONIN] REMOVE ALL
//
//    Private functions
//

static std::string get_string_aliased_arch_lowercase(tt::ARCH arch) {
    switch (arch) {
    case tt::ARCH::GRAYSKULL: 
        return "grayskull"; 
        break;
    case tt::ARCH::WORMHOLE: 
        return "wormhole"; 
        break;
    case tt::ARCH::WORMHOLE_B0: 
        return "wormhole"; 
        break;
    case tt::ARCH::BLACKHOLE: 
        return "blackhole"; 
        break;
    default: 
        return "invalid"; 
        break;
    }
}

static void build_failure(
        const string &target_name, 
        const string &op, 
        const string &cmd, 
        const string &log_file) {
    log_info(tt::LogBuildKernels, "{} {} failure -- cmd: {}", target_name, op, cmd);
    string cat = "cat " + log_file;
    if (fs::exists(log_file)) {
        // XXXX PGK(TODO) not portable
        if (system(cat.c_str())) {
            TT_THROW("Failed system comand {}", cat);
        }
    }
    TT_THROW("{} build failed", target_name);
}

//
//    JitBuildEnv
//

JitBuildEnv::JitBuildEnv() { }

void JitBuildEnv::init(uint32_t build_key, tt::ARCH arch) {
    // Paths
    this->root_ = llrt::OptionsG.get_root_dir();
    this->out_root_ = this->root_ + "built/";
    this->arch_ = arch;
    this->arch_name_ = get_string_lowercase(arch);
    this->aliased_arch_name_ = get_string_aliased_arch_lowercase(arch);

    this->out_firmware_root_ = this->out_root_ + to_string(build_key) + "/firmware/";
    this->out_kernel_root_ = this->out_root_ + to_string(build_key) + "/kernels/";

    // Tools
    this->gpp_ = this->root_ + "tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++ ";
    this->objcopy_ = this->root_ + "tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-objcopy ";

    // Flags
    string common_flags;
    switch (arch) {
    case ARCH::GRAYSKULL: 
        common_flags = "-mgrayskull -march=rv32iy -mtune=rvtt-b1 -mabi=ilp32 "; 
        break;
    case ARCH::WORMHOLE_B0: 
        common_flags = "-mwormhole -march=rv32imw -mtune=rvtt-b1 -mabi=ilp32 "; 
        break;
    case ARCH::BLACKHOLE: 
        common_flags = "-mblackhole -march=rv32iml -mtune=rvtt-b1 -mabi=ilp32 "; 
        break;
    default: 
        TT_ASSERT(false, "Invalid arch"); 
        break;
    }
    common_flags += "-std=c++17 -flto -ffast-math ";

    if (tt::llrt::OptionsG.get_riscv_debug_info_enabled()) {
        common_flags += "-g ";
    }

    this->cflags_ = common_flags;
    this->cflags_ +=
        "-fno-use-cxa-atexit -fno-exceptions "
        "-Wall -Werror -Wno-unknown-pragmas "
        "-Wno-error=multistatement-macros -Wno-error=parentheses "
        "-Wno-error=unused-but-set-variable -Wno-unused-variable "
        "-Wno-unused-function ";

    // Defines
    switch (arch) {
    case ARCH::GRAYSKULL: 
        this->defines_ = "-DARCH_GRAYSKULL "; 
        break;
    case ARCH::WORMHOLE_B0: 
        this->defines_ = "-DARCH_WORMHOLE "; 
        break;
    case ARCH::BLACKHOLE: 
        this->defines_ = "-DARCH_BLACKHOLE "; 
        break;
    default: 
        break;
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

#if 0 // [RONIN]
    if (tt::tt_metal::getDeviceProfilerState()) {
        if (tt::llrt::OptionsG.get_profiler_do_dispatch_cores()) {
            //TODO(MO): Standard bit mask for device side profiler options
            this->defines_ += "-DPROFILE_KERNEL=2 ";
        } else {
            this->defines_ += "-DPROFILE_KERNEL=1 ";
        }
    }
#endif

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        this->defines_ += "-DWATCHER_ENABLED ";
    }
    if (tt::llrt::OptionsG.get_watcher_noinline()) {
        this->defines_ += "-DWATCHER_NOINLINE ";
    }
    for (auto &feature: tt::llrt::OptionsG.get_watcher_disabled_features()) {
        this->defines_ += "-DWATCHER_DISABLE_" + feature + " ";
    }

    if (tt::llrt::OptionsG.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        this->defines_ += "-DDEBUG_PRINT_ENABLED ";
    }

    if (tt::llrt::OptionsG.get_kernels_nullified()) {
        this->defines_ += "-DDEBUG_NULL_KERNELS ";
    }

    if (tt::llrt::OptionsG.get_watcher_debug_delay()) {
        this->defines_ += 
            "-DWATCHER_DEBUG_DELAY=" + to_string(tt::llrt::OptionsG.get_watcher_debug_delay()) + " ";
    }

    // Includes
    // TODO(pgk) this list is insane
    this->includes_ = 
        string("") + "-I. " + "-I.. " + "-I" + this->root_ + " " + "-I" + this->root_ + "tt_metal " +
        "-I" + this->root_ + "tt_metal/include " + "-I" + this->root_ + "tt_metal/hw/inc " + "-I" +
        this->root_ + "tt_metal/hw/inc/debug " + "-I" + this->root_ + "tt_metal/hw/inc/" +
        this->aliased_arch_name_ + " " + "-I" + this->root_ + "tt_metal/hw/inc/" +
        this->aliased_arch_name_ + "/" + this->arch_name_ + "_defines " + "-I" + this->root_ +
        "tt_metal/hw/inc/" + this->aliased_arch_name_ + "/noc " + "-I" + this->root_ +
        "tt_metal/third_party/umd/device/" + this->arch_name_ + " " +  // TODO(fixme)
        "-I" + this->root_ + "tt_metal/hw/ckernels/" + this->arch_name_ + "/metal/common " + "-I" +
        this->root_ + "tt_metal/hw/ckernels/" + this->arch_name_ + "/metal/llk_io " + "-I" + this->root_ +
        "tt_metal/third_party/tt_llk_" + this->arch_name_ +
        "/common/inc " +  // TODO(fixme) datamovement fw shouldn't read this
        "-I" + this->root_ + "tt_metal/third_party/tt_llk_" + this->arch_name_ + "/llk_lib ";

    this->lflags_ = common_flags;
    this->lflags_ += 
        "-fno-exceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
}

//
//    JitBuildState
//

JitBuildState::JitBuildState(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw):
            env_(env), 
            core_id_(which), 
            is_fw_(is_fw) { }

// Fill in common state derived from the default state set up in the constructors
void JitBuildState::finish_init() {
    if (this->is_fw_) {
        this->defines_ += "-DFW_BUILD ";
    } else {
        this->defines_ += "-DKERNEL_BUILD ";
    }

    // Create the objs from the srcs
    for (string src: srcs_) {
        // Lop off the right side from the last "."
        string stub = src.substr(0, src.find_last_of("."));
        // Lop off the leading path
        stub = stub.substr(stub.find_last_of("/") + 1, stub.length());
        this->objs_.push_back(stub + ".o");
    }

    // Prepend root path to srcs, but not to outputs (objs) due to device dependency
    for (string &src: this->srcs_) {
        src = env_.root_ + src;
    }

    // Create list of object files for link
    for (const string &obj: this->objs_) {
        this->link_objs_ += obj + " ";
    }

    // Note the preceding slash which defies convention as this gets appended to
    // the kernel name used as a path which doesn't have a slash
    this->target_full_path_ = "/" + this->target_name_ + "/" + this->target_name_ + ".hex";
}

void JitBuildState::pre_compile(const string &kernel_in_path, const string &op_out_path) const { }

void JitBuildState::copy_kernel(
        const string &kernel_in_path, const string &op_out_path) const {
    // TODO(pgk): get rid of this copy, compile kernel file in place as its own .o
    const string out_dir = this->out_path_ + op_out_path + this->target_name_;
    const string dst = out_dir + "/kernel.cpp";
    // Assume kernel_in_path is absolute and test if it exists, if it doesn't exist then assume
    // it's relative to TT_METAL_HOME.
    const string src = 
        fs::exists(kernel_in_path) ? 
            kernel_in_path : 
            env_.get_root_path() + kernel_in_path;
    fs::copy(src, dst, fs::copy_options::overwrite_existing);
}

void JitBuildState::compile_one(
        const string &log_file,
        const string &out_dir,
        const JitBuildSettings *settings,
        const string &src,
        const string &obj) const {
    fs::create_directories(out_dir);

    // Add kernel specific defines
    string defines = this->defines_;
    if (settings != nullptr) {
        if (process_defines_at_compile) {
            settings->process_defines([&defines](const string &define, const string &value) {
                defines += "-D" + define + "=" + value + " ";
            });
        }

        settings->process_compile_time_args([&defines](int i, uint32_t value) {
            defines += "-DKERNEL_COMPILE_TIME_ARG_" + to_string(i) + "=" + to_string(value) + " ";
        });
    }

    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.gpp_;
    cmd += this->cflags_;
    cmd += defines;
    cmd += this->includes_;
    cmd += "-c -o " + obj + " " + src;

    log_debug(tt::LogBuildKernels, "    g++ compile cmd: {}", cmd);

    if (tt::llrt::OptionsG.get_watcher_enabled() && settings) {
        log_kernel_defines_and_args(out_dir, settings->get_full_kernel_name(), defines);
    }

    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "compile", cmd, log_file);
    }
}

#if 0 // [RONIN]
void JitBuildState::compile(const string &log_file, const string &out_dir, const JitBuildSettings *settings) const {
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < this->srcs_.size(); ++i) {
        launch_build_step(
            [this, &log_file, &out_dir, settings, i] {
                this->compile_one(log_file, out_dir, settings, this->srcs_[i], this->objs_[i]);
            },
            events);
    }

    sync_build_step(events);
    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        dump_kernel_defines_and_args(env_.get_out_kernel_root_path());
    }
}
#endif

void JitBuildState::compile(
        const string &log_file, 
        const string &out_dir, 
        const JitBuildSettings *settings) const {
    for (size_t i = 0; i < this->srcs_.size(); ++i) {
        launch_build_step(
            [this, &log_file, &out_dir, settings, i] {
                this->compile_one(log_file, out_dir, settings, this->srcs_[i], this->objs_[i]);
            });
    }

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        dump_kernel_defines_and_args(env_.get_out_kernel_root_path());
    }
}

void JitBuildState::link(const string &log_file, const string &out_dir) const {
    string lflags = this->lflags_;
    if (tt::llrt::OptionsG.get_build_map_enabled()) {
        lflags += " -Wl,-Map=" + out_dir + "linker.map";
    }

    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.gpp_;
    cmd += this->lflags_;
    cmd += this->link_objs_;

    if (!this->is_fw_) {
        string weakened_elf_name =
            env_.out_firmware_root_ + this->target_name_ + "/" + this->target_name_ + "_weakened.elf";
        cmd += " -Xlinker \"--just-symbols=" + weakened_elf_name + "\" ";
    }

    cmd += "-o " + out_dir + this->target_name_ + ".elf";
    log_debug(tt::LogBuildKernels, "    g++ link cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "link", cmd, log_file);
    }
}

void JitBuildState::elf_to_hex8(const string &log_file, const string &out_dir) const {
    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.objcopy_;
    cmd += " -O verilog " + this->target_name_ + ".elf" + " " + this->target_name_ + ".hex.tmp";

    log_debug(tt::LogBuildKernels, "    objcopy cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "objcopy", cmd, log_file);
    }
}

void JitBuildState::hex8_to_hex32(const string &log_file, const string &out_dir) const {
    ZoneScoped;
    auto write_data = [](std::ofstream &outf, std::vector<uint64_t> &data, uint64_t &ptr) {
        if (!data.empty()) {
            outf << "@" << std::setfill('0') << std::setw(8) << std::hex << (ptr >> 2) << "\n";
            for (size_t i = 0; i < data.size(); i += 4) {
                for (int j = 3; j >= 0; --j) {
                    if (i + j < data.size()) {
                        outf << std::setfill('0') << std::setw(2) << std::hex << data[i + j];
                    }
                }
                outf << "\n";
            }
        }
        data.clear();
    };

    auto pad_zeroes = [](std::vector<uint64_t> &data, uint32_t num) {
        for (unsigned int i = 0; i < num; i++) {
            data.push_back(0);
        }
    };

    std::ifstream inf(out_dir + this->target_name_ + ".hex.tmp");
    std::ofstream outf(out_dir + this->target_name_ + ".hex");
    std::string line;
    std::vector<uint64_t> data;
    uint64_t ptr = 0;

    while (std::getline(inf, line)) {
        if (line[0] == '@') {
            uint64_t addr = std::stol(line.substr(1), nullptr, 16);
            if (addr > ptr + 4) {
                write_data(outf, data, ptr);
                ptr = addr;
                pad_zeroes(data, (ptr % 4));
                ptr -= ptr % 4;
            } else {
                pad_zeroes(data, (addr - ptr - data.size()));
            }
        } else {
            std::istringstream iss(line);
            std::string tok;
            while (iss >> tok) {
                data.push_back(std::stol(tok, nullptr, 16));
            }
        }
    }
    write_data(outf, data, ptr);
}

// Given this elf (A) and a later elf (B):
// weakens symbols in A so that it can be used as a "library" for B.
// B imports A's weakened symbols, B's symbols of the
// same name don't result in duplicate symbols but B can reference A's symbols.
// Force the fw_export symbols to remain
// strong so to propogate link addresses
void JitBuildState::weaken(const string &log_file, const string &out_dir) const {
    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.objcopy_;
    cmd += 
        " --wildcard --weaken-symbol \"*\" --weaken-symbol \"!__fw_export_*\" " + 
        this->target_name_ + ".elf " +
        this->target_name_ + "_weakened.elf";

    log_debug(tt::LogBuildKernels, "    objcopy cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "objcopy weaken", cmd, log_file);
    }
}

#if 0 // [RONIN]
void JitBuildState::extract_zone_src_locations(const string &log_file) const {
    static std::atomic<bool> new_log = true;
    if (tt::tt_metal::getDeviceProfilerState()) {
        if (new_log.exchange(false) && std::filesystem::exists(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            std::remove(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG.c_str());
        }

        if (!std::filesystem::exists(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            tt::utils::create_file(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG);
        }

        // Only interested in log entries with KERNEL_PROFILER inside them as device code
        // tags source location info with it using pragma messages
        string cmd = "cat " + log_file + " | grep KERNEL_PROFILER";
        tt::utils::run_command(cmd, tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG, false);
    }
}
#endif

void JitBuildState::build(const JitBuildSettings *settings) const {
    string out_dir = 
        (settings == nullptr) ? 
            this->out_path_ + this->target_name_ + "/" : 
            this->out_path_ + settings->get_full_kernel_name() + this->target_name_ + "/";

    string log_file = out_dir + "build.log";
    if (fs::exists(log_file)) {
        std::remove(log_file.c_str());
    }

    compile(log_file, out_dir, settings);
    link(log_file, out_dir);
    elf_to_hex8(log_file, out_dir);
    hex8_to_hex32(log_file, out_dir);
    if (this->is_fw_) {
        weaken(log_file, out_dir);
    }

#if 0 // [RONIN]
    extract_zone_src_locations(log_file);
#endif
}

//
//    JitBuildDataMovement
//

JitBuildDataMovement::JitBuildDataMovement(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw) :
            JitBuildState(env, which, is_fw) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid data movement processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    // don't use memcpy for cpy loops
    this->cflags_ = env_.cflags_ + "-Os " + "-fno-tree-loop-distribute-patterns ";  
    this->includes_ = 
        env_.includes_ + "-I " + env_.root_ + "tt_metal/hw/firmware/src " + "-I " + env_.root_ +
        "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " + "-I " + env_.root_ +
        "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io ";

    this->defines_ = env_.defines_;

    // TODO(pgk): build these once at init into built/libs!
    this->srcs_.push_back("tt_metal/hw/toolchain/substitutes.cpp");

    this->lflags_ = env_.lflags_ + "-Os ";

    switch (this->core_id_) {
    case 0:
        this->target_name_ = "brisc";

        this->defines_ += "-DCOMPILE_FOR_BRISC ";

        this->srcs_.push_back("tt_metal/hw/firmware/src/tdma_xmov.c");
        this->srcs_.push_back("tt_metal/hw/firmware/src/" + env_.aliased_arch_name_ + "/noc.c");
        if (this->is_fw_) {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/brisc.cc");
        } else {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/brisck.cc");
        }

        this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/brisc.ld ";

        break;

    case 1:
        this->target_name_ = "ncrisc";

        this->defines_ += "-DCOMPILE_FOR_NCRISC ";

        if (this->is_fw_) {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/ncrisc.cc");
            this->srcs_.push_back("tt_metal/hw/toolchain/ncrisc-halt.S");
        } else {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/ncrisck.cc");
        }

        this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/ncrisc.ld ";

        break;
    }

    this->process_defines_at_compile = true;

    finish_init();
}

void JitBuildDataMovement::pre_compile(
        const string &kernel_in_path, const string &op_out_path) const {
    copy_kernel(kernel_in_path, op_out_path);
}

//
//    JitBuildCompute
//

JitBuildCompute::JitBuildCompute(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw): 
            JitBuildState(env, which, is_fw) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 3, "Invalid compute processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->cflags_ = env_.cflags_ + "-O3 ";

    this->defines_ = env_.defines_;

    this->includes_ = 
        env_.includes_ + "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/inc " + "-I" +
        env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " + "-I" + env_.root_ +
        "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io " + "-I" + env_.root_ +
        "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_api " + "-I" + env_.root_ +
        "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_api/llk_sfpu " + "-I" + env_.root_ +
        "tt_metal/third_party/sfpi/include " + "-I" + env_.root_ + "tt_metal/hw/firmware/src " + "-I" +
        env_.root_ + "tt_metal/third_party/tt_llk_" + env.arch_name_ + "/llk_lib ";

    this->srcs_.push_back("tt_metal/hw/toolchain/substitutes.cpp");
    if (this->is_fw_) {
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisc.cc");
        this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0.S");
    } else {
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisck.cc");
        this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
    }

    this->lflags_ = env_.lflags_ + "-O3 ";

    switch (this->core_id_) {
    case 0:
        this->target_name_ = "trisc0";

        this->defines_ += "-DUCK_CHLKC_UNPACK ";
        this->defines_ += "-DNAMESPACE=chlkc_unpack ";
        this->defines_ += "-DCOMPILE_FOR_TRISC=0 ";

        this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/trisc0.ld ";

        break;

    case 1:
        this->target_name_ = "trisc1";

        this->defines_ += "-DUCK_CHLKC_MATH ";
        this->defines_ += "-DNAMESPACE=chlkc_math ";
        this->defines_ += "-DCOMPILE_FOR_TRISC=1 ";

        this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/trisc1.ld ";

        break;

    case 2:
        this->target_name_ = "trisc2";

        this->defines_ += "-DUCK_CHLKC_PACK ";
        this->defines_ += "-DNAMESPACE=chlkc_pack ";
        this->defines_ += "-DCOMPILE_FOR_TRISC=2 ";

        this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/trisc2.ld ";

        break;
    }

    this->process_defines_at_compile = false;

    finish_init();
}

//
//    JitBuildEthernet
//

JitBuildEthernet::JitBuildEthernet(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw): 
            JitBuildState(env, which, is_fw) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid ethernet processor");
    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->includes_ = 
        env_.includes_ + "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ +
        "/metal/common " + "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ +
        "/metal/llk_io ";

    this->defines_ = env_.defines_;

    switch (this->core_id_) {
    case 0: 
        {
            this->target_name_ = "erisc";
            this->cflags_ = env_.cflags_ + "-Os -fno-delete-null-pointer-checks ";

            this->defines_ +=
                "-DCOMPILE_FOR_ERISC "
                "-DERISC "
                "-DRISC_B0_HW ";
            if (this->is_fw_) {
                this->defines_ += "-DLOADING_NOC=0 ";
            }

            this->includes_ += "-I " + env_.root_ + "tt_metal/hw/inc/ethernet ";

            this->srcs_.push_back("tt_metal/hw/toolchain/substitutes.cpp");
            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisc.cc");
                this->srcs_.push_back("tt_metal/hw/toolchain/erisc-early-exit.S");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisck.cc");
                this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
            }

            string linker_str;
            if (this->is_fw_) {
                linker_str = "tt_metal/hw/toolchain/erisc-b0-app.ld ";
            } else {
                linker_str = "tt_metal/hw/toolchain/erisc-b0-kernel.ld ";
            }
            this->lflags_ = 
                env_.lflags_ +
                "-Os "
                "-L" +
                env_.root_ +
                "/tt_metal/hw/toolchain "
                "-T" +
                env_.root_ + linker_str;
        }
        break;

    case 1:
        this->target_name_ = "idle_erisc";

        // don't use memcpy for cpy loops
        this->cflags_ = env_.cflags_ + "-Os " + "-fno-tree-loop-distribute-patterns ";  

        this->defines_ +=
            "-DCOMPILE_FOR_IDLE_ERISC "
            "-DERISC "
            "-DRISC_B0_HW ";

        this->includes_ += "-I " + env_.root_ + "tt_metal/hw/firmware/src ";

        // TODO(pgk): build these once at init into built/libs!
        this->srcs_.push_back("tt_metal/hw/toolchain/substitutes.cpp");
        this->srcs_.push_back("tt_metal/hw/firmware/src/" + env_.aliased_arch_name_ + "/noc.c");
        if (this->is_fw_) {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/idle_erisc.cc");
        } else {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/idle_erisck.cc");
        }
        this->lflags_ = env_.lflags_ + "-Os ";
        this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/idle-erisc.ld ";
        break;
    }

    this->process_defines_at_compile = true;

    finish_init();
}

void JitBuildEthernet::pre_compile(
        const string &kernel_in_path, const string &op_out_path) const {
    copy_kernel(kernel_in_path, op_out_path);
}

//
//    Public functions
//

void jit_build(
        const JitBuildState &build, 
        const JitBuildSettings *settings, 
        const string &kernel_in_path) {
    ZoneScoped;

    if (settings != nullptr) {
        build.pre_compile(kernel_in_path, settings->get_full_kernel_name());
    }

    build.build(settings);
}

#if 0 // [RONIN]
void jit_build_set(const JitBuildStateSet &build_set, const JitBuildSettings *settings, const string &kernel_in_path) {
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < build_set.size(); ++i) {
        // Capture the necessary objects by reference
        auto &build = build_set[i];
        launch_build_step(
            [build, settings, &kernel_in_path] {
                if (settings != nullptr) {
                    build->pre_compile(kernel_in_path, settings->get_full_kernel_name());
                }
                build->build(settings);
            },
            events);
    }
    sync_build_step(events);
}

void jit_build_subset(
    const JitBuildStateSubset &build_subset, const JitBuildSettings *settings, const string &kernel_in_path) {
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < build_subset.size; ++i) {
        // Capture the necessary objects by reference
        auto &build = build_subset.build_ptr[i];
        launch_build_step(
            [build, settings, &kernel_in_path] {
                if (settings != nullptr) {
                    build->pre_compile(kernel_in_path, settings->get_full_kernel_name());
                }
                build->build(settings);
            },
            events);
    }
    sync_build_step(events);
}
#endif

void jit_build_set(
        const JitBuildStateSet &build_set, 
        const JitBuildSettings *settings, 
        const string &kernel_in_path) {
    for (size_t i = 0; i < build_set.size(); ++i) {
        // Capture the necessary objects by reference
        auto &build = build_set[i];
        launch_build_step(
            [build, settings, &kernel_in_path] {
                if (settings != nullptr) {
                    build->pre_compile(kernel_in_path, settings->get_full_kernel_name());
                }
                build->build(settings);
            });
    }
}

void jit_build_subset(
        const JitBuildStateSubset &build_subset, 
        const JitBuildSettings *settings, 
        const string &kernel_in_path) {
    for (size_t i = 0; i < build_subset.size; ++i) {
        // Capture the necessary objects by reference
        auto &build = build_subset.build_ptr[i];
        launch_build_step(
            [build, settings, &kernel_in_path] {
                if (settings != nullptr) {
                    build->pre_compile(kernel_in_path, settings->get_full_kernel_name());
                }
                build->build(settings);
            });
    }
}
#endif    // [RONIN] REMOVE ALL

//
//    [RONIN] Fully refactored version
//

namespace {

using DeviceKernelBuilder = tt::metal::device::KernelBuilder;

std::unique_ptr<DeviceKernelBuilder> g_kernel_builder(DeviceKernelBuilder::create());

// TODO: Use environment variables?
std::string g_cpp_cmd_base = "clang++ -c -O3 --target=riscv32 -nostdinc";        

std::string get_string_aliased_arch_lowercase(tt::ARCH arch) {
    switch (arch) {
    case tt::ARCH::GRAYSKULL: 
        return "grayskull"; 
        break;
    case tt::ARCH::WORMHOLE: 
        return "wormhole"; 
        break;
    case tt::ARCH::WORMHOLE_B0: 
        return "wormhole"; 
        break;
    case tt::ARCH::BLACKHOLE: 
        return "blackhole"; 
        break;
    default: 
        return "invalid"; 
        break;
    }
}

std::string quote_define_value(const std::string &value) {
    // TODO: Replace this temporary placeholder with correct implementation
    bool must_quote = (value.find(" ") != std::string::npos);
    if (must_quote) {
        return "\"" + value + "\"";
    } else {
        return value;
    }
}

uint32_t get_kernel_device_addr(JitBuildState::TargetId id) {
    uint32_t addr = 0x0;
    switch (id) {
    case JitBuildState::TargetId::NCRISC: 
        addr = MEM_NCRISC_INIT_IRAM_L1_BASE;
        break;
    case JitBuildState::TargetId::BRISC: 
        addr = MEM_BRISC_FIRMWARE_BASE;
        break;
    case JitBuildState::TargetId::TRISC0: 
        addr = MEM_TRISC0_BASE;
        break;
    case JitBuildState::TargetId::TRISC1: 
        addr = MEM_TRISC1_BASE;
        break;
    case JitBuildState::TargetId::TRISC2: 
        addr = MEM_TRISC2_BASE;
        break;
    case JitBuildState::TargetId::ERISC:
        // TODO: Figure out correct value
        addr = 0x0;
        break;
    case JitBuildState::TargetId::IDLE_ERISC:
        // TODO: Figure out correct value
        addr = 0x0;
        break;
    default: 
        TT_ASSERT(false); 
        break;
    }
    return addr;
}

void create_hex_file(
        const std::string &path, 
        JitBuildState::TargetId id,
        const std::vector<uint8_t> &code, 
        uint32_t start_pc) {
    uint32_t addr = get_kernel_device_addr(id);
    int size = int(code.size());
    int words = size / 4;
    int tail = size % 4;
    const uint8_t *ptr = code.data();

    std::ofstream out(path);
    out << "@" << std::hex << "0x" << (addr / 4) << std::endl;
    out << std::hex << "0x" << start_pc << std::endl;
    for (int i = 0; i < words; i++) {
        out << std::hex << "0x" << 
            std::setw(2) << std::setfill('0') << int(ptr[4 * i + 3]) << 
            std::setw(2) << std::setfill('0') << int(ptr[4 * i + 2]) << 
            std::setw(2) << std::setfill('0') << int(ptr[4 * i + 1]) << 
            std::setw(2) << std::setfill('0') << int(ptr[4 * i]) << std::endl;
    }
    if (tail != 0) {
        uint8_t buf[4];
        for (int i = 0; i < 4; i++) {
            buf[i] = (i < tail) ? ptr[words * 4 + i] : 0;
        }
        out << std::hex << "0x" <<
            std::setw(2) << std::setfill('0') << int(buf[3]) << 
            std::setw(2) << std::setfill('0') << int(buf[2]) << 
            std::setw(2) << std::setfill('0') << int(buf[1]) << 
            std::setw(2) << std::setfill('0') << int(buf[0]) << std::endl;
    }
}

void create_null_hex_file(const std::string &path, JitBuildState::TargetId id) {
    uint32_t addr = get_kernel_device_addr(id);
    std::ofstream out(path);
    out << "@" << std::hex << "0x" << (addr / 4) << std::endl;
    out << "0x0" << std::endl;
}

} // namespace

//
//    JitBuildEnv
//

JitBuildEnv::JitBuildEnv() { }

void JitBuildEnv::init(uint32_t build_key, tt::ARCH arch) {
    // Paths
    this->root_ = llrt::OptionsG.get_root_dir();
    this->out_root_ = this->root_ + "built/";
    this->arch_ = arch;
    this->arch_name_ = get_string_lowercase(arch);
    this->aliased_arch_name_ = get_string_aliased_arch_lowercase(arch);

    this->out_firmware_root_ = this->out_root_ + std::to_string(build_key) + "/firmware/";
    this->out_kernel_root_ = this->out_root_ + std::to_string(build_key) + "/kernels/";

    // SKIPPED: this->gpp_
    // SKIPPED: this->objcopy_
    // SKIPPED: this->cflags_

    // Defines
    switch (arch) {
    case ARCH::GRAYSKULL: 
        this->defines_ = "-DARCH_GRAYSKULL "; 
        break;
    case ARCH::WORMHOLE_B0: 
        this->defines_ = "-DARCH_WORMHOLE "; 
        break;
    case ARCH::BLACKHOLE: 
        this->defines_ = "-DARCH_BLACKHOLE "; 
        break;
    default: 
        break;
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

    // SKIPPED: this->includes_
    // SKIPPED: this->lflags_
}

//
//    JitBuildState
//

JitBuildState::JitBuildState(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw):
            env_(env), 
            core_id_(which), 
            is_fw_(is_fw) { }

// Fill in common state derived from the default state set up in the constructors
void JitBuildState::finish_init() {
    if (this->is_fw_) {
        this->defines_ += "-DFW_BUILD ";
    } else {
        this->defines_ += "-DKERNEL_BUILD ";
    }

    // SKIPPED: this->link_objs_

    // Note the preceding slash which defies convention as this gets appended to
    // the kernel name used as a path which doesn't have a slash
    this->target_full_path_ = "/" + this->target_name_ + "/" + this->target_name_ + ".hex";
}

// SKIPPED: JitBuildState::pre_compile
// SKIPPED: JitBuildState::copy_kernel
// SKIPPED: JitBuildState::compile_one
// SKIPPED: JitBuildState::compile
// SKIPPED: JitBuildState::link
// SKIPPED: JitBuildState::elf_to_hex8
// SKIPPED: JitBuildState::hex8_to_hex32
// SKIPPED: JitBuildState::weaken

void JitBuildState::build(
        const JitBuildSettings *settings, 
        const std::string &kernel_in_path) const {
    std::string out_dir = 
        (settings == nullptr) ? 
            this->out_path_ + this->target_name_ + "/" : 
            this->out_path_ + settings->get_full_kernel_name() + this->target_name_ + "/";

    fs::create_directories(out_dir);

    // formula borrowed from original "hex8_to_hex32"
    // TODO: Check consistency with "target_full_path_" defined above
    //     Clients (e.g., llrt) will get_target_out_path(kernel_name)
    //     that is based on "target_full_path_"
    std::string target_out_path = out_dir + this->target_name_ + ".hex";

    if (this->is_fw_) {
        create_null_hex_file(target_out_path, this->target_id_);
    } else {
        // should work as target name is already encoded in "out_dir"
        std::string temp_dir = out_dir;
        build_kernel(
            settings, 
            kernel_in_path, 
            target_out_path,
            temp_dir);
    }
}

void JitBuildState::build_kernel(
        const JitBuildSettings *settings, 
        const std::string &kernel_in_path, 
        const std::string &target_out_path,
        const std::string &temp_dir) const {
    std::string defines = make_compiler_defines(settings);

    std::string root_path = env_.get_root_path();
    std::string include_path = root_path + "tt_metal/emulator/kernels";
    std::string cpp_cmd_base = 
        g_cpp_cmd_base + " -I " + root_path + " -I " + include_path;

    g_kernel_builder->configure(
        cpp_cmd_base,
        {}, // prefix_map
        "", // src_base_dir
        temp_dir);

    // Assume kernel_in_path is absolute and test if it exists,
    // if it doesn't exist then assume it's relative to TT_METAL_HOME.
    std::string kernel_file_path = 
        fs::exists(kernel_in_path) ? 
            kernel_in_path : 
            root_path + kernel_in_path;
    uint32_t code_base = get_kernel_device_addr(this->target_id_);
    bool is_compute = 
        (target_id_ == TargetId::TRISC0 ||
            target_id_ == TargetId::TRISC1 ||
            target_id_ == TargetId::TRISC2);
    std::vector<uint8_t> code;
    uint32_t start_pc;

    // make space for start_pc slot
    code_base += 4;

    g_kernel_builder->build(
        kernel_file_path,
        is_compute,
        defines,
        code_base,
        code, 
        start_pc);

    create_hex_file(target_out_path, this->target_id_, code, start_pc);
}

std::string JitBuildState::make_compiler_defines(const JitBuildSettings *settings) const {
    // Add kernel specific defines
    std::string defines = this->defines_;
    if (settings != nullptr) {
        settings->process_defines([&defines](const std::string &define, const std::string &value) {
            if (!value.empty()) {
                defines += " -D" + define + "=" + quote_define_value(value);
            } else {
                defines += " -D" + define;
            }
        });

        settings->process_compile_time_args([&defines](int i, uint32_t value) {
            defines += 
                " -DKERNEL_COMPILE_TIME_ARG_" + std::to_string(i) + 
                    "=" + std::to_string(value);
        });
    }
    return defines;
}

//
//    JitBuildDataMovement
//

JitBuildDataMovement::JitBuildDataMovement(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw) :
            JitBuildState(env, which, is_fw) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid data movement processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->defines_ = env_.defines_;

    switch (this->core_id_) {
    case 0:
        this->target_id_ = TargetId::BRISC;
        this->target_name_ = "brisc";
        this->defines_ += "-DCOMPILE_FOR_BRISC ";
        break;

    case 1:
        this->target_id_ = TargetId::NCRISC;
        this->target_name_ = "ncrisc";
        this->defines_ += "-DCOMPILE_FOR_NCRISC ";
        break;
    }

    // SKIPPED: this->process_defines_at_compile

    finish_init();
}

// SKIPPED: JitBuildDataMovement::pre_compile

//
//    JitBuildCompute
//

JitBuildCompute::JitBuildCompute(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw): 
            JitBuildState(env, which, is_fw) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 3, "Invalid compute processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->defines_ = env_.defines_;

    switch (this->core_id_) {
    case 0:
        this->target_id_ = TargetId::TRISC0;
        this->target_name_ = "trisc0";
        this->defines_ += "-DUCK_CHLKC_UNPACK ";
#if 0 // TODO: Revise this
        this->defines_ += "-DNAMESPACE=chlkc_unpack ";
#endif
        this->defines_ += "-DCOMPILE_FOR_TRISC=0 ";
        break;

    case 1:
        this->target_id_ = TargetId::TRISC1;
        this->target_name_ = "trisc1";
        this->defines_ += "-DUCK_CHLKC_MATH ";
#if 0 // TODO: Revise this
        this->defines_ += "-DNAMESPACE=chlkc_math ";
#endif
        this->defines_ += "-DCOMPILE_FOR_TRISC=1 ";
        break;

    case 2:
        this->target_id_ = TargetId::TRISC2;
        this->target_name_ = "trisc2";
        this->defines_ += "-DUCK_CHLKC_PACK ";
#if 0 // TODO: Revise this
        this->defines_ += "-DNAMESPACE=chlkc_pack ";
#endif
        this->defines_ += "-DCOMPILE_FOR_TRISC=2 ";
        break;
    }

    // SKIPPED: this->process_defines_at_compile

    finish_init();
}

//
//    JitBuildEthernet
//

JitBuildEthernet::JitBuildEthernet(
        const JitBuildEnv &env, 
        int which, 
        bool is_fw): 
            JitBuildState(env, which, is_fw) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid ethernet processor");
    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->defines_ = env_.defines_;

    switch (this->core_id_) {
    case 0: 
        this->target_id_ = TargetId::ERISC;
        this->target_name_ = "erisc";
        this->defines_ +=
            "-DCOMPILE_FOR_ERISC "
            "-DERISC "
            "-DRISC_B0_HW ";
        if (this->is_fw_) {
            this->defines_ += "-DLOADING_NOC=0 ";
        }
        break;

    case 1:
        this->target_id_ = TargetId::IDLE_ERISC;
        this->target_name_ = "idle_erisc";
        this->defines_ +=
            "-DCOMPILE_FOR_IDLE_ERISC "
            "-DERISC "
            "-DRISC_B0_HW ";
        break;
    }

    // SKIPPED: this->process_defines_at_compile

    finish_init();
}

// SKIPPED: JitBuildEthernet::pre_compile

//
//    Public functions
//

void jit_build(
        const JitBuildState &build, 
        const JitBuildSettings *settings, 
        const std::string &kernel_in_path) {
    ZoneScoped;
    build.build(settings, kernel_in_path);
}

void jit_build_set(
        const JitBuildStateSet &build_set, 
        const JitBuildSettings *settings, 
        const std::string &kernel_in_path) {
    for (size_t i = 0; i < build_set.size(); ++i) {
        // Capture the necessary objects by reference
        auto build = build_set[i];
        build->build(settings, kernel_in_path);
    }
}

void jit_build_subset(
        const JitBuildStateSubset &build_subset, 
        const JitBuildSettings *settings, 
        const std::string &kernel_in_path) {
    for (size_t i = 0; i < build_subset.size; ++i) {
        // Capture the necessary objects by reference
        auto build = build_subset.build_ptr[i];
        build->build(settings, kernel_in_path);
    }
}

}  // namespace tt::tt_metal

