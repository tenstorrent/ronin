// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/core.h>

#include <boost/core/demangle.hpp>
#if 0 // [RONIN]
#include <experimental/type_traits>
#else
#include "tt_stl/type_traits.hpp"
#endif
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {
namespace stl {

template <typename T>
std::string get_type_name() {
    return boost::core::demangle(typeid(T).name());
}

template <typename T>
std::string get_type_name(const T& object) {
    return get_type_name<T>();
}

// Forward Declare hash_object
namespace hash {

constexpr bool DEBUG_HASH_OBJECT_FUNCTION = false;

using hash_t = std::uint64_t;
constexpr hash_t DEFAULT_SEED = 1234;

namespace detail {

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>&) noexcept;

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>&) noexcept;

template <typename... Ts>
inline hash_t hash_object(const std::reference_wrapper<Ts...>&) noexcept;

template <typename T>
inline hash_t hash_object(const T&) noexcept;

template <typename... Types>
inline hash_t hash_objects(hash_t, const Types&...) noexcept;

}  // namespace detail

}  // namespace hash

namespace reflection {

using AttributeName = std::variant<const char*, std::string>;

struct Attribute final {
    static constexpr std::size_t ALIGNMENT = 32;
    using storage_t = std::array<std::byte, 256>;

    const std::string to_string() const { return this->implementations.to_string_impl_(this->type_erased_storage); }
    const std::size_t to_hash() const { return this->implementations.to_hash_impl_(this->type_erased_storage); }

    template <typename Type, typename BaseType = std::decay_t<Type>>
    Attribute(Type&& object) :
        pointer{new(&type_erased_storage) BaseType{std::forward<Type>(object)}},
        delete_storage{[](storage_t& self) { reinterpret_cast<BaseType*>(&self)->~BaseType(); }},
        copy_storage{[](storage_t& self, const void* other) -> void* {
            if constexpr (std::is_copy_constructible_v<BaseType>) {
                return new (&self) BaseType{*reinterpret_cast<const BaseType*>(other)};
            } else {
                static_assert(tt::stl::concepts::always_false_v<BaseType>);
            }
        }},
        move_storage{[](storage_t& self, void* other) -> void* {
            if constexpr (std::is_move_constructible_v<BaseType>) {
                return new (&self) BaseType{*reinterpret_cast<BaseType*>(other)};
            } else {
                static_assert(tt::stl::concepts::always_false_v<BaseType>);
            }
        }},

        implementations{
            .to_string_impl_ = [](const storage_t& storage) -> const std::string {
                const auto& object = *reinterpret_cast<const BaseType*>(&storage);
                return fmt::format("{}", object);
            },
            .to_hash_impl_ = [](const storage_t& storage) -> const std::size_t {
                const auto& object = *reinterpret_cast<const BaseType*>(&storage);
                return hash::detail::hash_object(object);
            }} {
        static_assert(sizeof(BaseType) <= sizeof(storage_t));
        static_assert(ALIGNMENT % alignof(BaseType) == 0);
    }

    void destruct() noexcept {
        if (this->pointer) {
            this->delete_storage(this->type_erased_storage);
        }
        this->pointer = nullptr;
    }

    Attribute(const Attribute& other) :
        pointer{other.pointer ? other.copy_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        copy_storage{other.copy_storage},
        move_storage{other.move_storage},
        implementations{other.implementations} {}

    Attribute& operator=(const Attribute& other) {
        if (other.pointer != this->pointer) {
            this->destruct();
            this->pointer = nullptr;
            if (other.pointer) {
                this->pointer = other.copy_storage(this->type_erased_storage, other.pointer);
            }
            this->delete_storage = other.delete_storage;
            this->copy_storage = other.copy_storage;
            this->move_storage = other.move_storage;
            this->implementations = other.implementations;
        }
        return *this;
    }

    Attribute(Attribute&& other) :
        pointer{other.pointer ? other.move_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        copy_storage{other.copy_storage},
        move_storage{other.move_storage},
        implementations{other.implementations} {}

    Attribute& operator=(Attribute&& other) {
        if (other.pointer != this->pointer) {
            this->destruct();
            this->pointer = nullptr;
            if (other.pointer) {
                this->pointer = other.move_storage(this->type_erased_storage, other.pointer);
            }
            this->delete_storage = other.delete_storage;
            this->copy_storage = other.copy_storage;
            this->move_storage = other.move_storage;
            this->implementations = other.implementations;
        }
        return *this;
    }

    ~Attribute() { this->destruct(); }

   private:
    alignas(ALIGNMENT) void* pointer = nullptr;
    alignas(ALIGNMENT) storage_t type_erased_storage;

    void (*delete_storage)(storage_t&) = nullptr;
    void* (*copy_storage)(storage_t& storage, const void*) = nullptr;
    void* (*move_storage)(storage_t& storage, void*) = nullptr;

    struct implementations_t {
        const std::string (*to_string_impl_)(const storage_t&) = nullptr;
        const std::size_t (*to_hash_impl_)(const storage_t&) = nullptr;
    };

    implementations_t implementations;
};

using Attributes = std::vector<std::tuple<AttributeName, Attribute>>;

namespace detail {
template <typename T>
using has_to_hash_t = decltype(std::declval<const T>().to_hash());

template <typename T>
constexpr bool supports_to_hash_v = std::experimental::is_detected_v<has_to_hash_t, T>;

template <typename T>
using has_to_string_t = decltype(std::declval<const T>().to_string());

template <typename T>
constexpr bool supports_to_string_v = std::experimental::is_detected_v<has_to_string_t, T>;

template <typename T>
using has_attributes_t = decltype(std::declval<T>().attributes());

template <typename T>
constexpr bool supports_runtime_time_attributes_v = std::experimental::is_detected_v<has_attributes_t, T>;

template <typename T>
static constexpr std::size_t get_num_attributes() {
    static_assert(
        std::tuple_size_v<decltype(T::attribute_names)> ==
            std::tuple_size_v<decltype(std::declval<T>().attribute_values())>,
        "Number of attribute_names must match number of attribute_values");
    return std::tuple_size_v<decltype(T::attribute_names)>;
}
template <typename T>
using has_attribute_names_t = decltype(std::declval<T>().attribute_names);

template <typename T>
using has_attribute_values_t = decltype(std::declval<T>().attribute_values());

template <typename T>
constexpr bool supports_compile_time_attributes_v = std::experimental::is_detected_v<has_attribute_names_t, T> and
                                                    std::experimental::is_detected_v<has_attribute_values_t, T>;

template <typename T>
constexpr bool supports_conversion_to_string_v =
    detail::supports_to_string_v<T> or detail::supports_compile_time_attributes_v<T> or
    detail::supports_runtime_time_attributes_v<T>;
}  // namespace detail

template <typename T>
Attributes get_attributes(const T& object) {
    if constexpr (tt::stl::reflection::detail::supports_compile_time_attributes_v<std::decay_t<T>>) {
        constexpr auto num_attributes = tt::stl::reflection::detail::get_num_attributes<std::decay_t<T>>();
        tt::stl::reflection::Attributes attributes;
        const auto attribute_values = object.attribute_values();
        [&object, &attributes, &attribute_values]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &attributes, &attribute_values] {
                    const auto& attribute_name = std::get<Ns>(object.attribute_names);
                    const auto& attribute = std::get<Ns>(attribute_values);
                    attributes.push_back({attribute_name, attribute});
                }(),
                ...);
        }(std::make_index_sequence<num_attributes>{});
        return attributes;
    } else if constexpr (tt::stl::reflection::detail::supports_runtime_time_attributes_v<std::decay_t<T>>) {
        return object.attributes();
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<T>, "Object doesn't support compile-time or run-time attributes!");
    }
}

static std::ostream& operator<<(std::ostream& os, const Attribute& attribute) {
    os << attribute.to_string();
    return os;
}

static std::ostream& operator<<(std::ostream& os, const Attributes& attributes) {
    os << "(";
    for (auto index = 0; index < attributes.size(); index++) {
        auto&& [key, value] = attributes[index];
        os << key;
        os << "=";
        os << value;
        if (index != attributes.size() - 1) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

template <typename T>
typename std::enable_if_t<detail::supports_conversion_to_string_v<T>, std::ostream>& operator<<(
    std::ostream& os, const T& object) {
    if constexpr (detail::supports_to_string_v<T>) {
        os << object.to_string();
    } else if constexpr (detail::supports_compile_time_attributes_v<T>) {
        constexpr auto num_attributes = detail::get_num_attributes<T>();
        os << get_type_name<T>();
        os << "(";

        if constexpr (num_attributes > 0) {
            const auto attribute_values = object.attribute_values();
            [&os, &object, &attribute_values]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                (
                    [&os, &object, &attribute_values] {
                        const auto& attribute = std::get<Ns>(attribute_values);
                        os << std::get<Ns>(object.attribute_names);
                        os << "=";
                        os << attribute;
                        os << ",";
                    }(),
                    ...);
            }(std::make_index_sequence<num_attributes - 1>{});

            const auto& attribute = std::get<num_attributes - 1>(attribute_values);
            os << std::get<num_attributes - 1>(object.attribute_names);
            os << "=";
            os << attribute;
        }

        os << ")";
    } else if constexpr (detail::supports_runtime_time_attributes_v<T>) {
        static_assert(std::is_same_v<decltype(object.attributes()), Attributes>);
        os << get_type_name<T>();
        os << object.attributes();
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Type cannot be converted to string");
    }
    return os;
}

template <typename T>
typename std::enable_if_t<std::is_enum<T>::value, std::ostream>& operator<<(std::ostream& os, const T& value) {
    os << magic_enum::enum_type_name<T>() << "::" << magic_enum::enum_name(value);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional) {
    if (optional.has_value()) {
        os << optional.value();
    } else {
        os << "std::nullopt";
    }
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::variant<Ts...>& variant) {
    std::visit([&os](const auto& value) { os << value; }, variant);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::reference_wrapper<T> reference_wrapper) {
    os << reference_wrapper.get();
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& tuple) {
    [&os, &tuple]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&os, &tuple]() { os << (Ns == 0 ? "" : ", ") << std::get<Ns>(tuple); }(), ...);
    }(std::make_index_sequence<sizeof...(Ts)>{});
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& array) {
    os << "{";
    for (auto index = 0; index < array.size(); index++) {
        const auto& element = array[index];
        os << element;
        if (index != array.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
    os << "{";
    for (auto index = 0; index < vector.size(); index++) {
        const auto& element = vector[index];
        os << element;
        if (index != vector.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set) {
    os << "{";
    auto index = 0;
    for (const auto& element : set) {
        os << element;
        if (index != set.size() - 1) {
            os << ", ";
        }
        index++;
    }
    os << "}";
    return os;
}

}  // namespace reflection
}  // namespace stl
}  // namespace tt

template <>
struct fmt::formatter<tt::stl::reflection::Attributes> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::stl::reflection::Attributes& attributes, format_context& ctx) const
        -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << attributes;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<tt::stl::reflection::detail::supports_conversion_to_string_v<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& object, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << object;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<std::is_enum<T>::value>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& value, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << value;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::optional<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::optional<T>& optional, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << optional;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename... Ts>
struct fmt::formatter<std::variant<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::variant<Ts...>& variant, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << variant;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::reference_wrapper<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::reference_wrapper<T> reference, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << reference;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename... Ts>
struct fmt::formatter<std::tuple<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::tuple<Ts...>& tuple, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << tuple;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T, std::size_t N>
struct fmt::formatter<std::array<T, N>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::array<T, N>& array, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << array;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::vector<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::vector<T>& vector, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::set<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::set<T>& set, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << set;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

namespace tt {
namespace stl {
namespace hash {

namespace detail {
template <typename T, typename = std::void_t<>>
struct is_std_hashable : std::false_type {};

template <typename T>
struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template <typename Test, template <typename...> class Ref>
constexpr bool is_specialization_v = is_specialization<Test, Ref>::value;

#if 0 // [RONIN]
template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>& array) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::array<{}, {}>: {}\n", get_type_name<T>(), N, array);
    }
    std::size_t hash = 0;
    [&array, &hash]<size_t... Ns>(std::index_sequence<Ns...>) {
        (
            [&array, &hash] {
                const auto& element = std::get<Ns>(array);
                hash = hash_objects(hash, element);
            }(),
            ...);
    }(std::make_index_sequence<N>{});
    return hash;
}
#else
template<typename T, std::size_t N, std::size_t Ns>
void hash_object_helper(const std::array<T, N>& array, std::size_t &hash) {
    const auto& element = std::get<Ns>(array);
    hash = hash_objects(hash, element);
}

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>& array) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::array<{}, {}>: {}\n", get_type_name<T>(), N, array);
    }
    std::size_t hash = 0;
    [&array, &hash]<size_t... Ns>(std::index_sequence<Ns...>) {
        (hash_object_helper<T, N, Ns>(array, hash), ...);
    }(std::make_index_sequence<N>{});
    return hash;
}
#endif

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>& variant) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::variant: {}\n", variant);
    }
    return std::visit([](const auto& value) { return hash_object(value); }, variant);
}

template <typename... Ts>
inline hash_t hash_object(const std::reference_wrapper<Ts...>& reference) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::reference_wrapper: {}\n", reference.get());
    }
    return hash_object(reference.get());
}

template <typename T>
inline hash_t hash_object(const T& object) noexcept {
    if constexpr (std::numeric_limits<T>::is_integer) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing integer of type {}: {}\n", get_type_name<T>(), object);
        }
        return object;
    } else if constexpr (detail::is_std_hashable_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing {} using std::hash: {}\n", get_type_name<T>(), object);
        }
        return std::hash<T>{}(object);
    } else if constexpr (std::is_same_v<T, tt::stl::reflection::Attributes>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing tt::stl::reflection::Attributes: {}\n", object);
        }
        auto hash = 0;
        for (auto&& [name, attribute] : object) {
            hash = hash_objects(hash, attribute);
        }
        return hash;
    } else if constexpr (tt::stl::reflection::detail::supports_to_hash_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using to_hash method: {}\n", get_type_name<T>(), object);
        }
        return object.to_hash();
    } else if constexpr (tt::stl::reflection::detail::supports_compile_time_attributes_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using compile-time attributes: {}\n", get_type_name<T>(), object);
        }
        constexpr auto num_attributes = reflection::detail::get_num_attributes<T>();
        std::size_t hash = 0;
        const auto attribute_values = object.attribute_values();
        [&object, &hash, &attribute_values]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &hash, &attribute_values] {
                    const auto& attribute = std::get<Ns>(attribute_values);
                    hash = hash_objects(hash, attribute);
                }(),
                ...);
        }(std::make_index_sequence<num_attributes>{});
        return hash;
    } else if constexpr (tt::stl::reflection::detail::supports_runtime_time_attributes_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using run-time attributes: {}\n", get_type_name<T>(), object);
        }
        return hash_objects(0, object.attributes());
    } else if constexpr (detail::is_specialization_v<T, std::vector>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::vector of type {}: {}\n", get_type_name<T>(), object);
        }
        auto hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (detail::is_specialization_v<T, std::set>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::set of type {}: {}\n", get_type_name<T>(), object);
        }
        auto hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (detail::is_specialization_v<T, std::optional>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::optional of type {}: {}\n", get_type_name<T>(), object);
        }
        if (object.has_value()) {
            return hash_object(object.value());
        } else {
            return 0;
        }
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Type doesn't support std::hash");
    }
}

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... args) noexcept {
    ([&seed](const auto& arg) { seed ^= hash_object(arg) + 0x9e3779b9 + (seed << 6) + (seed >> 2); }(args), ...);
    return seed;
}

}  // namespace detail

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... args) noexcept {
    return detail::hash_objects(seed, args...);
}

template <typename... Types>
inline hash_t hash_objects_with_default_seed(const Types&... args) noexcept {
    return detail::hash_objects(DEFAULT_SEED, args...);
}

}  // namespace hash
}  // namespace stl
}  // namespace tt
