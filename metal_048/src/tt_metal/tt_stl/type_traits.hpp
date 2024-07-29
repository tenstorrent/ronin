#pragma once

#include <type_traits>

namespace std {
namespace experimental {

struct nonesuch {
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

namespace detail {

template<class Default, class AlwaysVoid, template<class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};
 
template<class Default, template<class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

} // namespace detail
 
template<template<class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;
 
template<template<class...> class Op, class... Args>
using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;
 
template<class Default, template<class...> class Op, class... Args>
using detected_or = detail::detector<Default, void, Op, Args...>;

// additional utilities

template< template<class...> class Op, class... Args >
constexpr bool is_detected_v = is_detected<Op, Args...>::value;

#if 0
template< template<class...> class Op, class... Args >
constexpr inline bool is_detected_v = is_detected<Op, Args...>::value;
#endif

template< class Default, template<class...> class Op, class... Args >
using detected_or_t = typename detected_or<Default, Op, Args...>::type;

template< class Expected, template<class...> class Op, class... Args >
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template< class Expected, template<class...> class Op, class... Args >
constexpr bool is_detected_exact_v =
    is_detected_exact<Expected, Op, Args...>::value;

#if 0
template< class Expected, template<class...> class Op, class... Args >
constexpr inline bool is_detected_exact_v =
    is_detected_exact<Expected, Op, Args...>::value;
#endif

template< class To, template<class...> class Op, class... Args >
using is_detected_convertible =
    std::is_convertible<detected_t<Op, Args...>, To>;

template< class To, template<class...> class Op, class... Args >
constexpr bool is_detected_convertible_v =
    is_detected_convertible<To, Op, Args...>::value;

#if 0
template< class To, template<class...> class Op, class... Args >
constexpr inline bool is_detected_convertible_v =
    is_detected_convertible<To, Op, Args...>::value;
#endif

} // namespace experimental
} // namespace std

