//===- bcir_expected.hpp - std::expected, or the part of it the library uses ----===//
//
// The hosted C++ library reports a failure as a value, never an exception (the port audit's rule): `std::expected`
// where the standard library has it (`__cpp_lib_expected`), else this minimal stand-in of the same shape, for a
// toolchain whose library predates it -- Clang 18 over libstdc++ 13 declares no `std::expected`. Only what the
// library uses: a value or an error, `has_value`, `operator bool`, `operator*`, `operator->`, `error`, `unexpected`.
//===----------------------------------------------------------------------===//
#ifndef BCIR_EXPECTED_HPP
#define BCIR_EXPECTED_HPP

#include <version>
#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#include <expected>
namespace bcir {
template <class T, class E> using expected = std::expected<T, E>;
template <class E> auto unexpected(E e) { return std::unexpected<E>(e); }
}  // namespace bcir
#else
#include <utility>
#include <variant>
namespace bcir {
template <class E> struct unexpected_t { E error; };
template <class E> unexpected_t<E> unexpected(E e) { return unexpected_t<E>{e}; }
template <class T, class E> class expected {
 public:
  expected(T value) : v_(std::in_place_index<0>, std::move(value)) {}
  expected(unexpected_t<E> e) : v_(std::in_place_index<1>, e.error) {}
  bool has_value() const noexcept { return v_.index() == 0; }
  explicit operator bool() const noexcept { return has_value(); }
  T &operator*() noexcept { return *std::get_if<0>(&v_); }
  const T &operator*() const noexcept { return *std::get_if<0>(&v_); }
  T *operator->() noexcept { return std::get_if<0>(&v_); }
  E error() const noexcept { return *std::get_if<1>(&v_); }
 private:
  std::variant<T, E> v_;
};
}  // namespace bcir
#endif

#endif  // BCIR_EXPECTED_HPP
