#ifndef XYLO_TENSOR_
#define XYLO_TENSOR_

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <numeric>
#include <random>
#include <span>

#include <sstream>
#include <xeno/exception.h>

namespace xylo {

std::default_random_engine &default_generator();

template <std::size_t N> class array {
public:
  array(std::initializer_list<std::size_t> l) {
    std::copy(l.begin(), l.end(), content_.begin());
  }
  template <std::ranges::range R> array(R &&r) {
    std::copy(r.begin(), r.end(), content_.begin());
  }
  using content_t = std::array<std::size_t, N>;

  const content_t &get_content() const { return content_; }

private:
  content_t content_;
};

class memory_blob {
public:
  explicit memory_blob(std::size_t size = 0, bool on_device = false);
  explicit memory_blob(float *addr, bool on_device = false);

  memory_blob(memory_blob &&other);
  ~memory_blob();

  void operator=(const memory_blob &other);
  void operator=(memory_blob &&other);

  void set_on_device() { u_.val |= on_device_mask; }
  bool on_device() const { return u_.val & on_device_mask; }
  void set_borrowed() { u_.val |= 1ull << borrowed_pos; }
  bool borrowed() const { return u_.val & borrowed_mask; }

  float *addr() const;

private:
  float *do_alloc(std::size_t size, bool on_device);

  static constexpr uint64_t on_device_pos = 63;
  static constexpr uint64_t borrowed_pos = 62;

  static constexpr uint64_t addr_mask = uint64_t(-1) & ~(0x3ull << 62);
  static constexpr uint64_t on_device_mask = 1ull << 63;
  static constexpr uint64_t borrowed_mask = 1ull << 62;

  union {
    float *addr;
    uint64_t val = 0;
  } u_;
}; // namespace xylo

template <std::size_t N> class tensor_view;

template <std::size_t N> class tensor {
public:
  explicit tensor(std::array<std::size_t, N> shape, bool on_device = false)
      : shape_(shape), memory_blob_{size(), on_device} {}
  explicit tensor(std::initializer_list<std::size_t> shape,
                  bool on_device = false)
      : memory_blob_(std::accumulate(shape.begin(), shape.end(), 1ull,
                                     std::multiplies<std::size_t>{}),
                     on_device) {
    // Might be better to wrap our own array here.
    std::copy(shape.begin(), shape.end(), shape_.begin());
  }

  tensor(const tensor<N> &other)
      : shape_{other.shape()}, memory_blob_{other.size(), other.on_device()} {
    std::copy(other.data(), other.data() + other.size(), data());
  }
  explicit tensor(const tensor_view<N> &view);

  std::size_t rank() const { return N; }
  std::size_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1ull,
                           std::multiplies<std::size_t>{});
  }

  float *data() const { return memory_blob_.addr(); }
  bool on_device() const { return memory_blob_.on_device(); }

  std::array<std::size_t, N> shape() const { return shape_; }

  tensor_view<N - 1> operator[](std::size_t i) const {
    std::array<std::size_t, N - 1> new_shape;
    std::copy(shape_.begin() + 1, shape_.end(), new_shape.begin());
    std::size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), 1ull,
                        std::multiplies<std::size_t>{});
    return tensor_view<N - 1>(data() + i * new_size, new_shape, on_device());
  }

private:
  std::array<std::size_t, N> shape_;
  memory_blob memory_blob_;
};

template <> class tensor<1> {
public:
  using value_type = float;
  using iterator = float *;
  using const_iterator = const float *;
  explicit tensor(std::array<std::size_t, 1> shape, bool on_device = false)
      : shape_(shape), memory_blob_{size(), on_device} {}
  explicit tensor(std::size_t size, bool on_device = false)
      : shape_{size}, memory_blob_{size, on_device} {}
  explicit tensor(std::initializer_list<std::size_t> shape,
                  bool on_device = false)
      : memory_blob_(std::accumulate(shape.begin(), shape.end(), 1ull,
                                     std::multiplies<std::size_t>{}),
                     on_device) {
    // Might be better to wrap our own array here.
    std::copy(shape.begin(), shape.end(), shape_.begin());
  }

  tensor(const tensor &other)
      : shape_{other.shape()}, memory_blob_{other.size(), other.on_device()} {
    std::copy(other.data(), other.data() + other.size(), data());
  }
  explicit tensor(tensor_view<1> view);

  void operator=(float val);
  void operator=(tensor_view<1> other);
  void operator=(const tensor<1> &other);

  std::size_t rank() const { return 1; }
  std::size_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1ull,
                           std::multiplies<std::size_t>{});
  }

  iterator begin() noexcept { return memory_blob_.addr(); }
  iterator end() noexcept { return memory_blob_.addr() + size(); }
  const_iterator begin() const noexcept { return memory_blob_.addr(); }
  const_iterator end() const noexcept { return memory_blob_.addr() + size(); }

  float *data() const { return memory_blob_.addr(); }
  bool on_device() const { return memory_blob_.on_device(); }

  std::array<std::size_t, 1> shape() const { return shape_; }

  float operator[](std::size_t i) const {
    // TODO: Has to be on CPU.
    return *(data() + i);
  }
  float &operator[](std::size_t i) {
    // TODO: Has to be on CPU.
    return *(data() + i);
  }

private:
  std::array<std::size_t, 1> shape_;
  memory_blob memory_blob_;
};

template <std::size_t N> class tensor_view {
public:
  tensor_view(const tensor<N> &t)
      : shape_(t.shape()), memory_blob_(t.data(), t.on_device()) {}
  tensor_view(tensor_view<N> &other)
      : shape_(other.shape()), memory_blob_(other.data(), other.on_device()) {}
  std::size_t rank() const { return N; }
  std::size_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1ull,
                           std::multiplies<std::size_t>{});
  }

  float *data() const { return memory_blob_.addr(); }
  bool on_device() const { return memory_blob_.on_device(); }

  std::array<std::size_t, N> shape() const { return shape_; }

  tensor_view<1> flatten() const;

  tensor_view<N - 1> operator[](std::size_t i) const {
    std::array<std::size_t, N - 1> new_shape;
    std::copy(shape_.begin() + 1, shape_.end(), new_shape.begin());
    std::size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), 1ull,
                        std::multiplies<std::size_t>{});
    return tensor_view<N - 1>(data() + i * new_size, new_shape, on_device());
  }

private:
  tensor_view(float *addr, const std::array<std::size_t, N> &shape,
              bool on_device)
      : shape_(shape), memory_blob_(addr, on_device) {}
  std::array<std::size_t, N> shape_;
  memory_blob memory_blob_;
  friend class tensor_view<1>;
  friend class tensor<N + 1>;
  friend class tensor_view<N + 1>;
};

// Matrix View
template <> class tensor_view<2> {
public:
  class iterator;
  using value_type = tensor<1>;

  tensor_view(const tensor<2> &m)
      : shape_(m.shape()), memory_blob_{m.data(), m.on_device()} {}
  tensor_view(const tensor_view<2> &other)
      : shape_(other.shape()), memory_blob_{other.data(), other.on_device()} {}

  std::size_t rank() const { return 2; }
  std::size_t size() const { return shape_[0] * shape_[1]; }

  float *data() const { return memory_blob_.addr(); }
  bool on_device() const { return memory_blob_.on_device(); }

  iterator begin() const noexcept;
  iterator end() const noexcept;

  std::array<std::size_t, 2> shape() const { return shape_; }
  std::size_t num_rows() const { return shape_[0]; }
  std::size_t num_cols() const { return shape_[1]; }

  tensor_view<1> operator[](std::size_t i) const;

  tensor_view<1> flatten() const;

private:
  tensor_view(float *addr, const std::array<std::size_t, 2> &shape,
              bool on_device)
      : shape_(shape), memory_blob_(addr, on_device) {}
  std::array<std::size_t, 2> shape_;
  const memory_blob memory_blob_;

  friend class tensor_view<1>;
  friend class tensor<3>;
  friend class tensor_view<3>;
};

// Vector
template <> class tensor_view<1> {
public:
  using value_type = float;
  using iterator = float *;
  using const_iterator = const float *;

  tensor_view(const tensor<1> &v)
      : shape_(v.shape()), memory_blob_{v.data(), v.on_device()} {}
  tensor_view(const tensor_view<1> &other)
      : shape_(other.shape()), memory_blob_{other.data(), other.on_device()} {}

  std::size_t rank() const { return 1; }
  std::size_t size() const { return shape_[0]; }

  float *data() const { return reinterpret_cast<float *>(memory_blob_.addr()); }
  bool borrowed() const { return memory_blob_.borrowed(); }
  bool on_device() const { return memory_blob_.on_device(); }

  iterator begin() noexcept { return memory_blob_.addr(); }
  iterator end() noexcept { return memory_blob_.addr() + size(); }
  const_iterator begin() const noexcept { return memory_blob_.addr(); }
  const_iterator end() const noexcept { return memory_blob_.addr() + size(); }

  float operator[](uint64_t i) const { // TODO: Has to be on CPU
    return *(begin() + i);
  }
  float &operator[](uint64_t i) { return *(begin() + i); }

  std::array<std::size_t, 1> shape() const { return shape_; }

  void operator=(float val);
  void operator=(const tensor_view<1> &v);

  float dot(tensor_view<1> v) const;
  float sum() const;
  float mean() const;
  float variance() const;
  float stddev() const;
  float coef_variance() const;
  float max() const;
  std::size_t argmax() const;

  void normal_distribution(float mean, float stddev);
  void uniform_distribution(float lower, float upper);

  tensor_view<1> slice(std::size_t pos, std::size_t size) const;

  tensor_view<1> flatten() const {
    tensor_view<1> result(*const_cast<tensor_view<1> *>(this));
    return result;
  }

  template <std::size_t N>
  tensor_view<N> fold(std::array<std::size_t, N> shape) const;
  tensor_view<2> fold(std::size_t num_rows, std::size_t num_cols) const;

private:
  tensor_view(float *addr, const std::array<std::size_t, 1> shape,
              bool on_device)
      : shape_(shape), memory_blob_(addr, on_device) {}
  std::array<std::size_t, 1> shape_;
  const memory_blob memory_blob_;

  template <std::size_t N> friend class tensor_view;
  template <std::size_t N> friend class tensor;
  friend tensor_view<1> borrow_vector(std::span<float> s, bool on_device);
};

inline tensor_view<1> borrow_vector(std::span<float> s,
                                    bool on_device = false) {
  return tensor_view<1>(s.data(), {s.size()}, on_device);
}

using vector = tensor<1>;
using matrix = tensor<2>;
using vector_view = tensor_view<1>;
using matrix_view = tensor_view<2>;

template <std::size_t N>
inline tensor_view<N>
vector_view::fold(std::array<std::size_t, N> shape) const {
  return tensor_view<N>{memory_blob_.addr(), shape, memory_blob_.on_device()};
}

inline matrix_view vector_view::fold(std::size_t num_rows,
                                     std::size_t num_cols) const {
  return matrix_view{memory_blob_.addr(),
                     std::array<std::size_t, 2>{num_rows, num_cols},
                     memory_blob_.on_device()};
}
inline vector_view matrix_view::flatten() const {
  return vector_view{memory_blob_.addr(), std::array<std::size_t, 1>{size()},
                     memory_blob_.on_device()};
}

template <std::size_t N> inline vector_view tensor_view<N>::flatten() const {
  return vector_view{memory_blob_.addr(), std::array<std::size_t, 1>{size()},
                     memory_blob_.on_device()};
}

class matrix_view::iterator {
public:
  vector_view operator*() { return m_[idx_]; }
  bool operator!=(const iterator &other) { return idx_ != other.idx_; }
  void operator++() { ++idx_; }

private:
  iterator(tensor_view<2> m, std::size_t idx) : m_(m), idx_(idx) {}
  matrix_view m_;
  std::size_t idx_;

  friend matrix_view;
};

// Immitating at&t assemply.
// Matrix
void transpose(matrix_view in, matrix_view out);
void matmul_transposed(matrix_view in1, const matrix_view in2, matrix_view out);
void matmul(matrix_view in1, matrix_view in2, matrix_view out);
void add(xylo::matrix_view in1, xylo::matrix_view in2, xylo::matrix_view out);
void minus(xylo::matrix_view in1, xylo::matrix_view in2, xylo::matrix_view out);
void multiply(xylo::matrix_view in1, xylo::matrix_view in2,
              xylo::matrix_view out);
void divide(xylo::matrix_view in1, xylo::matrix_view in2,
            xylo::matrix_view out);

// Vector
void add(xylo::vector_view in1, xylo::vector_view in2, xylo::vector_view out);
void minus(xylo::vector_view in1, xylo::vector_view in2, xylo::vector_view out);
void multiply(xylo::vector_view in1, xylo::vector_view in2,
              xylo::vector_view out);
void divide(xylo::vector_view in1, xylo::vector_view in2,
            xylo::vector_view out);
void abs(xylo::vector_view in, xylo::vector_view out);
void sin(xylo::vector_view in, xylo::vector_view out);
void exp(xylo::vector_view in, xylo::vector_view out);
void log(xylo::vector_view in, xylo::vector_view out);
void sqrt(xylo::vector_view in, xylo::vector_view out);

// We take these functions outside of classes to take advanage of view
// auto-conversion.

// TODO: Get rid of the class-level fold and flatten.
template <std::size_t N> tensor_view<N> view(tensor<N> t) {
  xylo::tensor_view<N> tv(t);
  return tv;
}
template <std::size_t N> vector_view flatten(tensor_view<N> t) {
  return t.flatten();
}
// TODO: investigate why the conversion constructor isn't working.
template <std::size_t N> vector_view flatten(const tensor<N> &t) {
  tensor_view<N> v(t);
  return v.flatten();
}

template <std::size_t N>
tensor_view<N> fold(vector_view v, const xylo::array<N> &shape) {
  return v.fold(shape.get_content());
}
inline vector_view slice(vector_view v, std::size_t pos, std::size_t size) {
  return v.slice(pos, size);
}
inline matrix_view slice(matrix_view v, std::size_t pos, std::size_t size) {
  std::size_t num_rows = v.num_rows();
  std::size_t num_cols = v.num_cols();
  vector_view vv = flatten(v);
  vector_view s = slice(vv, pos * num_cols, (pos + size) * num_cols);
  return fold<2>(s, {size, num_cols});
}

// This has to be here, because it's a template.
template <std::size_t N>
tensor<N>::tensor(const tensor_view<N> &tv)
    : shape_{tv.shape()}, memory_blob_{tv.size(), tv.on_device()} {
  std::copy(tv.data(), tv.data() + tv.size(), data());
}

} // namespace xylo

xylo::matrix transpose(xylo::matrix_view m);
xylo::matrix matmul_transposed(xylo::matrix_view in1, xylo::matrix_view in2);
xylo::matrix matmul(xylo::matrix_view in1, xylo::matrix_view in2);

void operator+=(xylo::matrix_view v, float scalar);
void operator+=(xylo::matrix_view v1, xylo::matrix_view v2);
void operator-=(xylo::matrix_view v, float scalar);
void operator-=(xylo::matrix_view v1, xylo::matrix_view v2);
void operator*=(xylo::matrix_view v, float scalar);
void operator*=(xylo::matrix_view v1, xylo::matrix_view v2);
void operator/=(xylo::matrix_view v, float scalar);
void operator/=(xylo::matrix_view v1, xylo::matrix_view v2);

xylo::matrix operator+(xylo::matrix_view v1, xylo::matrix_view v2);
xylo::matrix operator-(xylo::matrix_view v1, xylo::matrix_view v2);
xylo::matrix operator*(xylo::matrix_view v1, xylo::matrix_view v2);
xylo::matrix operator/(xylo::matrix_view v1, xylo::matrix_view v2);

void operator+=(xylo::vector_view v, float scalar);
void operator+=(xylo::vector_view v1, xylo::vector_view v2);
void operator-=(xylo::vector_view v, float scalar);
void operator-=(xylo::vector_view v1, xylo::vector_view v2);
void operator*=(xylo::vector_view v, float scalar);
void operator*=(xylo::vector_view v1, xylo::vector_view v2);
void operator/=(xylo::vector_view v, float scalar);
void operator/=(xylo::vector_view v1, xylo::vector_view v2);

float dot(xylo::vector_view v1, xylo::vector_view v2);
float sum(xylo::vector_view v);
float mean(xylo::vector_view v);
float variance(xylo::vector_view v);
float stddev(xylo::vector_view v);
float coef_variance(xylo::vector_view v);
float max(xylo::vector_view v);

std::size_t argmax(xylo::vector_view v);
std::size_t discrete_distribution(xylo::vector_view v);

void normal_distribution(float mean, float stddev, xylo::vector_view v);
void uniform_distribution(float lower, float higher, xylo::vector_view v);

bool operator==(xylo::vector_view v1, xylo::vector_view v2);
xylo::vector operator+(xylo::vector_view v1, xylo::vector_view v2);
xylo::vector operator+(xylo::vector_view v, float scalar);
xylo::vector operator-(xylo::vector_view v1, xylo::vector_view v2);
xylo::vector operator-(xylo::vector_view v, float scalar);
xylo::vector operator*(xylo::vector_view v1, xylo::vector_view v2);
xylo::vector operator*(xylo::vector_view v, float scalar);
xylo::vector operator/(xylo::vector_view v1, xylo::vector_view v2);
xylo::vector operator/(xylo::vector_view v, float scalar);
xylo::vector abs(xylo::vector_view v);
xylo::vector sin(xylo::vector_view v);
xylo::vector exp(xylo::vector_view v);
xylo::vector log(xylo::vector_view v);
xylo::vector sqrt(xylo::vector_view v);

namespace xeno::string {
template <std::size_t N> std::string streamable(const xylo::tensor<N> &t) {
  return streamable(xylo::tensor_view<N>(t));
}

inline std::string streamable(xylo::vector_view v) {
  std::stringstream out;
  bool first = true;
  out << "[";
  for (float f : v) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << f;
  }
  out << "]";
  return out.str();
}

inline std::string streamable(xylo::matrix_view m) {
  std::stringstream out;
  bool first = true;
  out << "[";
  for (xylo::vector_view v : m) {
    if (!first) {
      out << '\n';
    }
    first = false;
    out << streamable(v);
  }
  out << "]";
  return out.str();
}
} // namespace xeno::string
#endif // XYLO_TENSOR_
