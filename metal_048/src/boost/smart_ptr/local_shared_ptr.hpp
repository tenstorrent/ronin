#pragma once

#include <memory>

namespace boost {

template<typename T>
using local_shared_ptr = std::shared_ptr<T>;

} // namespace boost

