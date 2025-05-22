// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "core/common.hpp"
#include "core/frontend.hpp"
#include "core/api.hpp"

namespace ronin {
namespace tanto {
namespace front {

bool run_frontend(
        const FrontendArgs &args, 
        const std::string &input_code,
        std::string &output_code,
        std::vector<std::string> &errors) {
    output_code.clear();
    errors.clear();
    Frontend frontend;
    for (std::pair<std::string, std::string> define: args.defines) {
        frontend.add_define(define.first, define.second);
    }
    for (std::pair<uint32_t, uint32_t> param: args.params) {
        frontend.add_param(param.first, param.second);
    }
    bool ok = frontend.compile(args.mode, input_code, output_code);
    if (!ok) {
        errors = frontend.get_errors();
        return false;
    }
    return true;
}

} // namespace front
} // namespace tanto
} // namespace ronin

