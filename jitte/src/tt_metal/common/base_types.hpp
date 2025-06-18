// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>

#include "constants.hpp"

enum class MathFidelity : uint8_t
{
    LoFi          = 0,
    HiFi2         = 2,
    HiFi3         = 3,
    HiFi4         = 4,
    Invalid       = 0xff,
};

inline std::ostream& operator<<(std::ostream& os, const MathFidelity &fidelity)
{
    switch (fidelity) {
        case MathFidelity::LoFi: os << "LoFi"; break;
        case MathFidelity::HiFi2: os << "HiFi2"; break;
        case MathFidelity::HiFi3: os << "HiFi3"; break;
        case MathFidelity::HiFi4: os << "HiFi4"; break;
        case MathFidelity::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown format");
    }
    return os;
}

template<>
struct std::hash<MathFidelity>
{
    std::size_t operator()(MathFidelity const& obj) const noexcept
    {
        return static_cast<std::size_t>(obj);
    }
};
