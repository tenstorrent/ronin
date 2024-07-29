#pragma once

#include <memory>

#include <boost/smart_ptr/local_shared_ptr.hpp>

namespace boost {

template<typename T, typename ...Args>
local_shared_ptr<T> make_local_shared(Args &&...args) {
    return std::make_shared<T, Args...>(std::forward<Args>(args)...);
}

} // namespace boost

