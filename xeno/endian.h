#ifndef XENO_ENDIAN_
#define XENO_ENDIAN_

#include <array>
#include <bit>
#include <span>

namespace xeno {
template <typename T>
constexpr std::array<std::byte, sizeof(T)> to_wire(const T &v) {
  using bytes_t = std::array<std::byte, sizeof(T)>;
  bytes_t result;
  const bytes_t *src = reinterpret_cast<const bytes_t *>(&v);

  if (std::endian::native == std::endian::big) {
    result = *src;
    return result;
  }

  if (std::endian::native == std::endian::little) {
    const std::size_t size = sizeof(T);
    for (int i = 0; i < size; ++i) {
      result[i] = (*src)[size - i - 1];
    }
    return result;
  }

  result = *src;
  return result;
}

template <typename T> constexpr T from_wire(std::span<const std::byte> src) {
  using bytes_t = std::array<std::byte, sizeof(T)>;
  T result;
  bytes_t *bytes = reinterpret_cast<bytes_t *>(&result);

  if (std::endian::native == std::endian::big) {
    std::copy(src.begin(), src.end(), bytes->begin());
    return result;
  }

  if (std::endian::native == std::endian::little) {
    const std::size_t size = sizeof(T);
    for (int i = 0; i < size; ++i) {
      (*bytes)[i] = src[size - i - 1];
    }
    return result;
  }
  return result;
}

template <typename T>
constexpr std::array<std::byte, sizeof(T)> &to_native(T &v) {
  auto *ptr = reinterpret_cast<std::array<std::byte, sizeof(T)> *>(&v);
  return *ptr;
}

template <typename T>
constexpr std::array<const std::byte, sizeof(T)> &to_native(const T &v) {
  auto *ptr = reinterpret_cast<std::array<const std::byte, sizeof(T)> *>(&v);
  return *ptr;
}

template <typename T>
constexpr const T &from_native(std::span<const std::byte> src) {
  auto *ptr = reinterpret_cast<const T *>(src.data());
  return *ptr;
}

} // namespace xeno

#endif // XENO_ENDIAN_
