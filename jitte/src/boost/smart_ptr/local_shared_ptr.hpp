// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace boost {

template<typename T>
using local_shared_ptr = std::shared_ptr<T>;

} // namespace boost

