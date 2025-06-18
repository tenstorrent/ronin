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

#include "jit_build/genfiles.hpp"
#include "jit_build/kernel_args.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"

#include "device/api/kernel_builder.hpp"

namespace fs = std::filesystem;

using namespace std;
using namespace tt;

namespace tt::tt_metal {

namespace {

using DeviceKernelBuilder = tt::metal::device::KernelBuilder;

std::unique_ptr<DeviceKernelBuilder> g_kernel_builder(DeviceKernelBuilder::create());

// TODO: Use environment variables?
std::string g_cpp_cmd_base = "clang++ -c -O3 --target=riscv32 -nostdinc";        

std::string get_string_aliased_arch_lowercase(tt::ARCH arch) {
    switch (arch) {
    case tt::ARCH::GRAYSKULL: 
        return "grayskull"; 
    case tt::ARCH::WORMHOLE: 
        return "wormhole"; 
    case tt::ARCH::WORMHOLE_B0: 
        return "wormhole"; 
    case tt::ARCH::BLACKHOLE: 
        return "blackhole"; 
    default: 
        return "invalid"; 
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
        addr = MEM_TRISC0_FIRMWARE_BASE;
        break;
    case JitBuildState::TargetId::TRISC1: 
        addr = MEM_TRISC1_FIRMWARE_BASE;
        break;
    case JitBuildState::TargetId::TRISC2: 
        addr = MEM_TRISC2_FIRMWARE_BASE;
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
        this->defines_ += "-DCOMPILE_FOR_TRISC=0 ";
        break;

    case 1:
        this->target_id_ = TargetId::TRISC1;
        this->target_name_ = "trisc1";
        this->defines_ += "-DUCK_CHLKC_MATH ";
        this->defines_ += "-DCOMPILE_FOR_TRISC=1 ";
        break;

    case 2:
        this->target_id_ = TargetId::TRISC2;
        this->target_name_ = "trisc2";
        this->defines_ += "-DUCK_CHLKC_PACK ";
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
