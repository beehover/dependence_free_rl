#ifndef XYLO_MATRIX_
#define XYLO_MATRIX_

#include <immintrin.h>
#include <malloc.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <vector>

namespace xylo {

template <typename T, std::size_t align = 256> class array {
public:
  using iterator = T *;
  using const_iterator = const T *;
  array() = default;
  array(std::size_t size) : size_(size) {
    data_ = reinterpret_cast<T *>(memalign(align, size * sizeof(T)));
  }
  array(std::span<const T> data) : size_(data.size()) {
    data_ = reinterpret_cast<T *>(memalign(align, size_ * sizeof(T)));
    std::copy(data.begin(), data.end(), data_);
  }
  ~array() { free(data_); }

  array(array &&other) { *this = std::move(other); }
  array(const array &other) { *this = other; }

  const array &operator=(array &&other) {
    std::swap(data_, other.data_);
    return *this;
  }

  const array &operator=(const array &other) {
    std::copy(other.begin(), other.end(), begin());
    return *this;
  }

  void resize(std::size_t size) {
    size_ = size;
    free(data_);
    data_ = reinterpret_cast<T *>(memalign(align, size * sizeof(T)));
  }

  T &operator[](std::size_t pos) { return data_[pos]; }
  const T &operator[](std::size_t pos) const { return data_[pos]; }

  iterator begin() { return data_; }
  iterator end() { return data_ + size_; }

  const_iterator begin() const { return data_; }
  const_iterator cbegin() const { return data_; }
  const_iterator end() const { return data_ + size_; }
  const_iterator cend() const { return data_ + size_; }

  T *data() { return data_; }
  const T *data() const { return data_; }

  std::size_t size() { return size_; }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
};

// Matrix
template <typename T> class matrix {
public:
  matrix() = default;

  matrix(std::span<const T> data, std::size_t num_cols)
      : num_rows_(data.size() / num_cols), num_cols_(num_cols),
        data_(data.size()) {
    std::copy(data.begin(), data.end(), data_.begin());
  }

  matrix(std::size_t num_rows, std::size_t num_cols)
      : num_rows_(num_rows), num_cols_(num_cols), data_(num_rows * num_cols) {}

  matrix(const matrix &other) {
    num_rows_ = other.num_rows_;
    num_cols_ = other.num_cols_;
    data_.resize(other.num_rows_ * other.num_cols_);
    std::copy(other.data_.begin(), other.data_.end(), data_.begin());
  }

  matrix(matrix &&other) {
    num_rows_ = other.num_rows_;
    num_cols_ = other.num_cols_;
    data_ = std::move(other.data_);
  }

  const matrix &operator=(const matrix &other) {
    num_rows_ = other.num_rows_;
    num_cols_ = other.num_cols_;
    data_.resize(other.num_rows_ * other.num_cols_);
    std::copy(other.data_.begin(), other.data_.end(), data_.begin());
    return *this;
  }

  const matrix &operator=(matrix &&other) {
    num_rows_ = other.num_rows_;
    num_cols_ = other.num_cols_;
    data_ = std::move(other.data_);
    return *this;
  }

  T *data() { return data_.data(); }
  const T *data() const { return data_.data(); }

  std::size_t num_rows() const { return num_rows_; }

  std::size_t num_cols() const { return num_rows_; }

#if 0
  std::vector<vector_slice<T>>
  block(std::pair<std::size_t, std::size_t> row_sel,
        std::pair<std::size_t, std::size_t> col_sel) {}
#endif

  void transpose() {
    if (num_rows_ != 1 && num_cols_ != 1) {
      array<T> v(num_rows_ * num_cols_);
      for (std::size_t i = 0; i < num_rows_; ++i) {
        for (std::size_t j = 0; j < num_cols_; ++j) {
          *(v.begin() + (j * num_rows_ + i)) =
              *(data_.begin() + (i * num_cols_ + j));
        }
      }
      data_ = std::move(v);
    }
    std::swap(num_rows_, num_cols_);
  }

  std::string debug_string() const {
    std::ostringstream os;
    for (std::size_t i = 0; i < num_rows_; ++i) {
      os << std::endl;
      os << '|';
      for (std::size_t j = 0; j < num_cols_; ++j) {
        os << data_[i * num_cols_ + j];
        if (j != num_cols_ - 1) {
          os << ' ';
        }
      }
      os << '|';
    }
    return os.str();
  }

private:
  std::size_t num_rows_;
  std::size_t num_cols_;
  array<T> data_;
};

template <typename T>
xylo::matrix<T> matmul_transposed(const xylo::matrix<T> &a,
                                  const xylo::matrix<T> &b) {
  xylo::matrix<T> result(a.num_rows(), b.num_rows());
  T *d_r = result.data();

  for (std::size_t i = 0; i < a.num_rows(); ++i) {
    for (std::size_t j = 0; j < b.num_rows(); ++j) {
      T sum = 0;
      const T *d_a = a.data(), *d_b = b.data();
      for (std::size_t k = 0; k < a.num_cols(); ++k) {
        sum += d_a[i * a.num_cols() + k] * d_b[j * b.num_cols() + k];
      }
      d_r[i * b.num_rows() + j] = sum;
    }
  }
  return result;
}

#if 0
xylo::matrix<float> matmul_transposed(const xylo::matrix<float> &a,
                                      const xylo::matrix<float> &b) {
  xylo::matrix<float> result(a.num_rows(), b.num_rows());
  float *d_r = result.data();
  const float *d_a = a.data(), *d_b = b.data();

  for (std::size_t i = 0; i < a.num_rows(); ++i) {
    for (std::size_t j = 0; j < b.num_rows(); ++j) {
      float sum = 0;
      for (std::size_t k = 0; k < a.num_cols(); k += 8) {
        alignas(256) float f[8];
        auto aw = _mm256_load_ps(d_a + (i * a.num_cols() + k));
        auto bw = _mm256_load_ps(d_b + (j * a.num_cols() + k));
        auto cw = _mm256_dp_ps(aw, bw, 0xf1);
        // auto cw = _mm256_mul_ps(aw, bw);
        _mm256_store_ps(f, cw);
        sum += f[0] + f[4];
      }
      d_r[i * b.num_rows() + j] = sum;
    }
  }
  return result;
}
#endif
} // namespace xylo

template <typename T>
xylo::matrix<T> operator*(const xylo::matrix<T> &a, const xylo::matrix<T> &b) {
  xylo::matrix<T> result(a.num_rows(), b.num_cols());
  const T *d_a = a.data(), *d_b = b.data();
  T *d_r = result.data();
  for (std::size_t i = 0; i < a.num_rows(); ++i) {
    for (std::size_t j = 0; j < b.num_cols(); ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < a.num_cols(); ++k) {
        sum += d_a[i * a.num_cols() + k] * d_b[k * b.num_cols() + j];
      }
      d_r[i * b.num_cols() + j] = sum;
    }
  }
  return result;
}

#endif // XYLO_MATRIX_
