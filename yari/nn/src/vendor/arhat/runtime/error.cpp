//
// Copyright (c) 2019-2025 FRAGATA COMPUTER SYSTEMS AG
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

#include <cstdio>
#include <cstdarg>
#include <string>

#include "runtime/arhat.hpp"

namespace arhat {

//
//    ArhatError
//

ArhatError::ArhatError(const std::string &msg):
        m_msg(msg) { }

ArhatError::~ArhatError() { }

const char *ArhatError::what() const noexcept {
    return m_msg.c_str();
}

//
//    Public functions
//

void Error(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    throw ArhatError(buf);
}

} // namespace arhat

