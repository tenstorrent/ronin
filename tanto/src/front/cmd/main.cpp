// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/util.hpp"
#include "core/api.hpp"

using namespace ronin::tanto::front;

namespace {

bool has_prefix(const std::string &str, const std::string &prefix) {
    return (str.substr(0, prefix.size()) == prefix);
}

bool is_ident(const std::string &src) {
    size_t n = src.size();
    for (size_t i = 0; i < n; i++) {
        char ch = src[i];
        if (!((ch >= 'A' && ch <= 'Z') ||
                (ch >= 'a' && ch <= 'z') ||
                (i > 0 && ch >= '0' && ch <= '9') ||
                ch == '_')) {
            return false;
        }
    }
    return true;
}

bool parse_uint32(const std::string &src, uint32_t &value) {
    errno = 0;
    char *end = nullptr;
    unsigned long temp = strtoul(src.c_str(), &end, 0);
    if (errno != 0) {
        errno = 0;
        return false;
    }
    value = uint32_t(temp);
    if ((unsigned long)value != temp) {
        value = 0;
        return false;
    }
    return true;
}

bool parse_mode(const std::string &src, FrontendMode &mode) {
    if (src == "read") {
        mode = FrontendMode::READ;
    } else if (src == "compute") {
        mode = FrontendMode::COMPUTE;
    } else if (src == "write") {
        mode = FrontendMode::WRITE;
    } else {
        mode = FrontendMode::UNDEF;
        return false;
    }
    return true;
}

bool parse_define(const std::string &src, std::string &name, std::string &value) {
    size_t sep = src.find('=');
    if (sep != std::string::npos) {
        name = src.substr(0, sep);
        value = src.substr(sep + 1);
    } else {
        name = src;
        value.clear();
    }
    if (!is_ident(name)) {
        return false;
    }
    return true;
}

bool parse_param(const std::string &src, uint32_t &index, uint32_t &value) {
    index = 0;
    value = 0;
    size_t sep = src.find('=');
    if (sep == std::string::npos) {
        return false;
    }
    if (!parse_uint32(src.substr(0, sep), index)) {
        return false;
    }
    if (!parse_uint32(src.substr(sep+1), value)) {
        return false;
    }
    return true;
}

bool validate_args(const FrontendArgs &args) {
    if (args.mode == FrontendMode::UNDEF) {
        printf("Missing frontend mode\n");
        return false;
    }
    return true;
}

bool parse_args(
        int argc,
        char **argv,
        FrontendArgs &args,
        std::string &input_path) {
    args.mode = FrontendMode::UNDEF;
    int iarg = 1;
    for ( ; iarg < argc; iarg++) {
        char *argp = argv[iarg];
        if (argp[0] != '-') {
            break;
        }
        if (has_prefix(argp, "--mode=")) {
            FrontendMode mode;
            if (!parse_mode(argp + 7, mode)) {
                printf("Invalid frontend mode\n");
                return false;
            }
            args.mode = mode;
        } else if (has_prefix(argp, "-D")) {
            std::string name;
            std::string value;
            if (!parse_define(argp + 2, name, value)) {
                printf("Invalid define\n");
                return false;
            }
            args.defines.emplace_back(name, value);
        } else if (has_prefix(argp, "-P")) {
            uint32_t index = 0;
            uint32_t value = 0;
            if (!parse_param(argp + 2, index, value)) {
                printf("Invalid parameter\n");
                return false;
            }
            args.params.emplace_back(index, value);
        }
    }
    if (!validate_args(args)) {
        return false;
    }
    if (iarg != argc - 1) {
        printf("Usage: tanto [<option> ...] <input_path>\n");
        return false;
    }
    input_path = argv[iarg];
    return true;
}

} // namespace

int main(int argc, char **argv) {
    FrontendArgs args;
    std::string input_path;
    if (!parse_args(argc, argv, args, input_path)) {
        printf("Invalid command line arguments\n");
        return 1;
    }
    std::string input_code;
    if (!read_file(input_path, input_code)) {
        printf("Cannot read input file [%s]\n", input_path.c_str());
        return 1;
    }
    std::string output_code;
    std::vector<std::string> errors;
    bool ok = run_frontend(args, input_code, output_code, errors);
    if (!ok) {
        for (std::string text: errors) {
            printf("%s\n", text.c_str());
        }
        printf("Compilation failed\n");
        return 1;
    }
    printf("%s\n", output_code.c_str());
    return 0;
}

