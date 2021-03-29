#include <algorithm>
#include <cmath>
#include <cstring>
#include <experimental/source_location>
#include <functional>
#include <immintrin.h>
#include <numeric>
#include <random>
#include <sstream>

#include <xeno/exception.h>
#include <xeno/string.h>
#include <xeno/time.h>
#include <xylo/tensor.h>

namespace xylo {

namespace {
float *default_alloc(std::size_t size) {
  if (size == 0)
    return nullptr;

  void *result;
  posix_memalign(&result, 32, size * sizeof(float));
  // lg() << "alloc'ing " << size << " bytes @" << result;
  return reinterpret_cast<float *>(result);
}

void default_dealloc(float *p) {
  // freeing a null ptr shouldn't be a big deal, but we bail early anyway.
  if (p == nullptr)
    return;

  // lg() << "freeing " << p;
  free(p);
}

float *gpu_alloc(std::size_t size) { return nullptr; }
void gpu_dealloc(float *p) {}

template <typename T1, typename T2>
void check_shape_equal(const T1 &t1, const T2 &t2,
                       std::experimental::source_location location =
                           std::experimental::source_location::current()) {
#if 1
  if (t1.shape() != t2.shape()) {
    throw xeno::error("different tensor shapes.", location);
  }
#endif
}

void check_transpose_shapes(const matrix_view m1, const matrix_view m2,
                            std::experimental::source_location location =
                                std::experimental::source_location::current()) {
  if (m1.num_cols() != m2.num_rows() || m1.num_rows() != m2.num_cols()) {
    throw xeno::error("wrong shapes for transpose", location);
  }
}
void check_matmul_shapes(const matrix_view m1, const matrix_view m2,
                         std::experimental::source_location location =
                             std::experimental::source_location::current()) {
  // Transposed
  if (m1.num_cols() != m2.num_cols()) {
    std::string message = xeno::string::strcat(
        "wrong shapes for matmul: ", m1.num_rows(), 'x', m1.num_cols(), " vs. ",
        m2.num_cols(), 'x', m2.num_rows());
    throw xeno::error(message, location);
  }
}

std::default_random_engine
    g_generator((xeno::time::now() - xeno::time::epoch()).time_.tv_sec);
} // namespace

std::default_random_engine &default_generator() { return g_generator; }

// ******** Memory blob methods ********
memory_blob::memory_blob(std::size_t size, bool on_device)
    : u_{on_device ? gpu_alloc(size) : default_alloc(size)} {
  static_assert(sizeof(float *) == sizeof(uint64_t));
  if (on_device) {
    set_on_device();
  }
}

memory_blob::memory_blob(float *addr, bool on_device) : u_{addr} {
  set_borrowed();
  if (on_device) {
    set_on_device();
  }
}
memory_blob::memory_blob(memory_blob &&other) { std::swap(u_, other.u_); }

memory_blob::~memory_blob() {
  if (borrowed())
    return;

  if (on_device())
    gpu_dealloc(addr());

  default_dealloc(addr());
}

void memory_blob::operator=(const memory_blob &other) { u_ = other.u_; }
void memory_blob::operator=(memory_blob &&other) { std::swap(u_, other.u_); }

float *memory_blob::addr() const {
  auto result = u_;
  result.val &= addr_mask;
  return result.addr;
}

float *do_alloc(std::size_t size, bool on_device) {
  if (size == 0)
    return nullptr;

  if (on_device)
    return gpu_alloc(size);

  return default_alloc(size);
}

// ******** Tensor methods ********
vector::tensor(vector_view view)
    : shape_{view.shape()}, memory_blob_{view.size(), view.on_device()} {
  std::ranges::copy(view, data());
}
void vector::operator=(float val) { std::ranges::fill(*this, val); }
void vector::operator=(vector_view other) {
  check_shape_equal(*this, other);
  std::ranges::copy(other, begin());
}
void vector::operator=(const vector &other) {
  check_shape_equal(*this, other);
  std::ranges::copy(other, begin());
}

// ******** Matrix View methods ********

matrix_view::iterator matrix_view::begin() const noexcept {
  return iterator(*this, 0);
}
matrix_view::iterator matrix_view::end() const noexcept {
  return iterator(*this, num_rows());
}

vector_view matrix_view::operator[](std::size_t i) const {
  return flatten().slice(i * num_cols(), num_cols());
}

// ******** Vector View methods ********
void vector_view::operator=(float val) { std::ranges::fill(*this, val); }
void vector_view::operator=(const vector_view &other) {
  check_shape_equal(*this, other);
  std::ranges::copy(other, begin());
}

float vector_view::dot(const vector_view other) const {
  check_shape_equal(*this, other);
  return std::inner_product(begin(), end(), other.begin(), 0.0f);
}

float vector_view::sum() const {
  if (size() == 0)
    return 0.0f;
  return std::accumulate(begin(), end(), 0.0f);
}
float vector_view::mean() const { return sum() / size(); }
float vector_view::variance() const {
  if (size() == 0)
    return 0.0f;
  float m = mean();
  auto variance_sum = [m](float sum, float elem) {
    float diff = elem - m;
    return sum + diff * diff;
  };
  return std::accumulate(begin(), end(), 0.0f, variance_sum) / size();
}
float vector_view::stddev() const { return ::sqrt(variance()); }
float vector_view::coef_variance() const {
  float m = mean();
  float sd = stddev();
  if (m == 0.0f && sd == 0.0f)
    return 0.0f;
  return mean() / stddev();
}
std::size_t vector_view::argmax() const {
  return std::ranges::distance(begin(), std::ranges::max_element(*this));
}

void vector_view::normal_distribution(float mean, float stddev) {
  std::normal_distribution<float> dist{mean, stddev};
  auto &gen = default_generator();
  std::ranges::for_each(*this, [&](float &val) { val = dist(gen); });
}
void vector_view::uniform_distribution(float lower, float upper) {
  std::uniform_real_distribution<float> dist{lower, upper};
  auto &gen = default_generator();
  std::ranges::for_each(*this, [&](float &val) { val = dist(gen); });
}

vector_view vector_view::slice(std::size_t pos, std::size_t size) const {
  vector_view other(memory_blob_.addr() + pos, std::array<std::size_t, 1>{size},
                    on_device());
  return other;
}

// ******** Global matrix functions ********
void transpose(const matrix_view in, matrix_view out) {
  check_transpose_shapes(in, out);
  for (std::size_t i = 0; i < in.num_rows(); ++i) {
    for (std::size_t j = 0; j < out.num_rows(); ++j) {
      out[j][i] = in[i][j];
    }
  }
}

void matmul_transposed(matrix_view in1, matrix_view in2, matrix_view out) {
  check_matmul_shapes(in1, in2);
  std::size_t m = in1.num_rows();
  std::size_t n = in2.num_rows();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out[i][j] = dot(in1[i], in2[j]);
    }
  }
}
void matmul(matrix_view in1, matrix_view in2, matrix_view out) {
  matmul_transposed(in1, ::transpose(in2), out);
}

void add(matrix_view in1, matrix_view in2, matrix_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  add(flatten(in1), flatten(in2), flatten(out));
}
void minus(matrix_view in1, matrix_view in2, matrix_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  minus(flatten(in1), flatten(in2), flatten(out));
}
void multiply(matrix_view in1, matrix_view in2, matrix_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  multiply(flatten(in1), flatten(in2), flatten(out));
}
void divide(matrix_view in1, matrix_view in2, matrix_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  divide(flatten(in1), flatten(in2), flatten(out));
}

// ******** Global vector functions ********
// We don't merge back into the vector class, because these functios offer
// us the possibilities of writing to the output directly, saving a copy.
void add(vector_view in1, vector_view in2, vector_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  std::ranges::transform(in1, in2, out.begin(), std::plus{});
}
void add(vector_view in, float scalar, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(in, out.begin(),
                         std::bind(std::plus{}, std::placeholders::_1, scalar));
}
void minus(vector_view in1, vector_view in2, vector_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  std::ranges::transform(in1, in2, out.begin(), std::minus{});
}
void minus(vector_view in, float scalar, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(
      in, out.begin(), std::bind(std::minus{}, std::placeholders::_1, scalar));
}
void multiply(vector_view in1, vector_view in2, vector_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  std::ranges::transform(in1, in2, out.begin(), std::multiplies{});
}
void multiply(vector_view in, float scalar, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(
      in, out.begin(),
      std::bind(std::multiplies{}, std::placeholders::_1, scalar));
}
void divide(vector_view in1, vector_view in2, vector_view out) {
  check_shape_equal(in1, in2);
  check_shape_equal(in1, out);
  std::ranges::transform(in1, in2, out.begin(), std::divides{});
}
void divide(vector_view in, float scalar, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(
      in, out.begin(),
      std::bind(std::divides{}, std::placeholders::_1, scalar));
}
void abs(vector_view in, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(in, out.begin(), ::fabsf);
}
void sin(vector_view in, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(in, out.begin(), ::sinf);
}
void exp(vector_view in, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(in, out.begin(), ::expf);
}
void log(vector_view in, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(in, out.begin(), ::logf);
}
void sqrt(vector_view in, vector_view out) {
  check_shape_equal(in, out);
  std::ranges::transform(in, out.begin(), ::sqrtf);
}

} // namespace xylo

// ******** Out-of-namespace matrix operators ********
xylo::matrix transpose(xylo::matrix_view in) {
  xylo::matrix out(std::array<std::size_t, 2>{in.num_cols(), in.num_rows()});
  xylo::transpose(in, out);
  return out;
}
xylo::matrix matmul_transposed(xylo::matrix_view in1, xylo::matrix_view in2) {
  xylo::matrix out(std::array<std::size_t, 2>{in1.num_rows(), in2.num_rows()});
  xylo::matmul_transposed(in1, in2, out);
  return out;
}
xylo::matrix matmul(xylo::matrix_view in1, xylo::matrix_view in2) {
  xylo::matrix out(std::array<std::size_t, 2>{in1.num_rows(), in2.num_cols()});
  xylo::matmul(in1, in2, out);
  return out;
}

void operator+=(xylo::matrix_view m, float val) { flatten(m) += val; }
void operator+=(xylo::matrix_view m1, xylo::matrix_view m2) {
  xylo::check_shape_equal(m1, m2);
  flatten(m1) += flatten(m2);
}
void operator-=(xylo::matrix_view m, float val) { flatten(m) -= val; }
void operator-=(xylo::matrix_view m1, xylo::matrix_view m2) {
  xylo::check_shape_equal(m1, m2);
  flatten(m1) -= flatten(m2);
}
void operator*=(xylo::matrix_view m, float val) { flatten(m) *= val; }
void operator*=(xylo::matrix_view m1, xylo::matrix_view m2) {
  xylo::check_shape_equal(m1, m2);
  flatten(m1) *= flatten(m2);
}
void operator/=(xylo::matrix_view m, float val) { flatten(m) /= val; }
void operator/=(xylo::matrix_view m1, xylo::matrix_view m2) {
  xylo::check_shape_equal(m1, m2);
  flatten(m1) /= flatten(m2);
}

xylo::matrix operator+(xylo::matrix_view in1, xylo::matrix_view in2) {
  xylo::matrix out(std::array<std::size_t, 2>{in1.num_rows(), in1.num_cols()});
  xylo::add(in1, in2, out);
  return out;
}
xylo::matrix operator-(xylo::matrix_view in1, xylo::matrix_view in2) {
  xylo::matrix out(std::array<std::size_t, 2>{in1.num_rows(), in1.num_cols()});
  xylo::minus(in1, in2, out);
  return out;
}

// ******** Out-of-namespace vector operators ********
void operator+=(xylo::vector_view v, float val) {
  std::ranges::transform(v, v.begin(), std::bind_front(std::plus{}, val));
}
void operator+=(xylo::vector_view v1, xylo::vector_view v2) {
  xylo::check_shape_equal(v1, v2);
  std::ranges::transform(v1, v2, v1.begin(), std::plus{});
}
void operator-=(xylo::vector_view v, float val) {
  std::ranges::transform(v, v.begin(), std::bind_front(std::minus{}, val));
}
void operator-=(xylo::vector_view v1, xylo::vector_view v2) {
  xylo::check_shape_equal(v1, v2);
  std::ranges::transform(v1, v2, v1.begin(), std::minus{});
}
void operator*=(xylo::vector_view v, float val) {
  std::ranges::transform(v, v.begin(), std::bind_front(std::multiplies{}, val));
}
void operator*=(xylo::vector_view v1, xylo::vector_view v2) {
  xylo::check_shape_equal(v1, v2);
  std::ranges::transform(v1, v2, v1.begin(), std::multiplies{});
}
void operator/=(xylo::vector_view v, float val) {
  std::ranges::transform(v, v.begin(), std::bind_front(std::divides{}, val));
}
void operator/=(xylo::vector_view v1, xylo::vector_view v2) {
  xylo::check_shape_equal(v1, v2);
  std::ranges::transform(v1, v2, v1.begin(), std::divides{});
}

float fastdot(xylo::vector_view v1, xylo::vector_view v2) {
  float sum = 0;
  std::size_t num_fast_words = v1.size() / 8;
  for (std::size_t i = 0; i < num_fast_words; ++i) {
    alignas(32) float f[8];
    auto aw = _mm256_load_ps(v1.data() + i * 8);
    auto bw = _mm256_load_ps(v2.data() + i * 8);
    auto cw = _mm256_dp_ps(aw, bw, 0xf1);
    _mm256_store_ps(f, cw);
    sum += f[0] + f[4];
  }

  std::size_t residual_offset = num_fast_words * 8;
  auto v1_residual = slice(v1, residual_offset, v1.size() - residual_offset);
  auto v2_residual = slice(v2, residual_offset, v2.size() - residual_offset);
  sum += std::inner_product(v1_residual.begin(), v1_residual.end(),
                            v2_residual.begin(), 0.0f);
  return sum;
}

bool use_fastdot(xylo::vector_view v1, xylo::vector_view v2) {
  std::size_t size = v1.size();
  // return (size >= 8) && (size % 8 == 0);
  return reinterpret_cast<uint64_t>(v1.data()) % 32 == 0 &&
         reinterpret_cast<uint64_t>(v2.data()) % 32 == 0;
}

float dot(xylo::vector_view v1, xylo::vector_view v2) {
  xylo::check_shape_equal(v1, v2);
  if (use_fastdot(v1, v2))
    return fastdot(v1, v2);
  return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0f);
}

float sum(xylo::vector_view v) {
  if (v.size() == 0)
    return 0.0f;
  return std::accumulate(v.begin(), v.end(), 0.0f);
}

float mean(xylo::vector_view v) { return sum(v) / v.size(); }

float variance(xylo::vector_view v) {
  if (v.size() == 0)
    return 0.0f;
  float m = mean(v);
  auto variance_sum = [m](float sum, float elem) {
    float diff = elem - m;
    return sum + diff * diff;
  };
  return std::accumulate(v.begin(), v.end(), 0.0f, variance_sum) / v.size();
}
float stddev(xylo::vector_view v) { return sqrt(variance(v)); }

float coef_variance(xylo::vector_view v) {
  float m = mean(v);
  float sd = stddev(v);
  if (m == 0.0f && sd == 0.0f)
    return 0.0f;
  return m / sd;
}

float max(xylo::vector_view v) { return std::ranges::max(v); }

std::size_t argmax(xylo::vector_view v) {
  return std::ranges::distance(v.begin(), std::ranges::max_element(v));
}
std::size_t discrete_distribution(xylo::vector_view v) {
  std::discrete_distribution<std::size_t> dist{v.begin(), v.end()};
  return dist(xylo::default_generator());
}

void normal_distribution(float mean, float stddev, xylo::vector_view v) {
  std::normal_distribution<float> dist{mean, stddev};
  auto &gen = xylo::default_generator();
  std::ranges::for_each(v, [&](float &val) { val = dist(gen); });
}
void uniform_distribution(float lower, float upper, xylo::vector_view v) {
  std::uniform_real_distribution<float> dist{lower, upper};
  auto &gen = xylo::default_generator();
  std::ranges::for_each(v, [&](float &val) { val = dist(gen); });
}

bool operator==(xylo::vector_view in1, xylo::vector_view in2) {
  if (in1.size() != in2.size())
    return false;

  if (in1.data() == in2.data())
    return true;

  return std::memcmp(in1.data(), in2.data(), in1.size() * sizeof(float)) == 0;
}

xylo::vector operator+(xylo::vector_view in1, xylo::vector_view in2) {
  xylo::vector out(std::array<std::size_t, 1>{in1.size()});
  xylo::add(in1, in2, out);
  return out;
}
xylo::vector operator+(xylo::vector_view in, float scalar) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::add(in, scalar, out);
  return out;
}
xylo::vector operator-(xylo::vector_view in1, xylo::vector_view in2) {
  xylo::vector out(std::array<std::size_t, 1>{in1.size()});
  xylo::minus(in1, in2, out);
  return out;
}
xylo::vector operator-(xylo::vector_view in, float scalar) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::minus(in, scalar, out);
  return out;
}
xylo::vector operator*(xylo::vector_view in1, xylo::vector_view in2) {
  xylo::vector out(std::array<std::size_t, 1>{in1.size()});
  xylo::multiply(in1, in2, out);
  return out;
}
xylo::vector operator*(xylo::vector_view in, float scalar) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::multiply(in, scalar, out);
  return out;
}
xylo::vector operator/(xylo::vector_view in1, xylo::vector_view in2) {
  xylo::vector out(std::array<std::size_t, 1>{in1.size()});
  xylo::divide(in1, in2, out);
  return out;
}
xylo::vector operator/(xylo::vector_view in, float scalar) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::divide(in, scalar, out);
  return out;
}
xylo::vector sin(xylo::vector_view in) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::sin(in, out);
  return out;
}
xylo::vector abs(xylo::vector_view in) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::abs(in, out);
  return out;
}
xylo::vector exp(xylo::vector_view in) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::exp(in, out);
  return out;
}
xylo::vector log(xylo::vector_view in) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::log(in, out);
  return out;
}
xylo::vector sqrt(xylo::vector_view in) {
  xylo::vector out(std::array<std::size_t, 1>{in.size()});
  xylo::sqrt(in, out);
  return out;
}
