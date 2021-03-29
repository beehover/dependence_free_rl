#ifndef XYLO_NN_
#define XYLO_NN_

#include <functional>

#include <memory>
#include <xeno/string.h>
#include <xylo/tensor.h>

namespace xylo {

void normal_initialize(std::size_t fan_in, vector_view v) {
  normal_distribution(0, 0.01, v);
}

void he_initialize(std::size_t fan_in, vector_view v) {
  normal_distribution(0, ::sqrtf(2.0f / fan_in), v);
}

class layer {
public:
  explicit layer(std::string_view name = "") : name_(name) {}
  virtual ~layer() = default;
  virtual matrix forward(matrix_view t) = 0;
  virtual matrix backward(matrix_view input, matrix_view loss) = 0;
  virtual vector gradient(matrix_view input, matrix_view backprop) = 0;
  virtual vector_view parameters() const = 0;

  std::string_view name() { return name_; }

protected:
  std::string name_;
};

namespace {

// Potentially move this into tensor.h
matrix pad(matrix_view m, std::size_t padded_size) {
  matrix result({m.num_rows(), padded_size});
  for (std::size_t i = 0; i < m.num_rows(); ++i) {
    slice(result[i], 0, m.num_cols()) = m[i];
    slice(result[i], 0, m.num_cols()) = m[i];
  }
  return result;
}
matrix depad(matrix_view m, std::size_t depadded_size) {
  matrix result({m.num_rows(), depadded_size});
  for (std::size_t i = 0; i < m.num_rows(); ++i) {
    result[i] = slice(m[i], 0, depadded_size);
  }
  return result;
}

} // namespace

// Parameters are arranged by rows. [a_0_0, a_0_1, a_0_2 ... a_0_n] is a
// node. Suppose there are m nodes, then at the end of the parameters there will
// be an extra [b_0, b_1, ..., b_m] row. When a batch comes in, it'll need to
// multiply with the transpose of the parameters.
class matmul_layer : public layer {
public:
  matmul_layer(std::size_t input_size, std::size_t output_size,
               std::string_view name = "")
      : layer(name), parameters_({(input_size + 1) * output_size}),
        a_(fold<2>(slice(parameters_, 0, input_size * output_size),
                   {output_size, input_size})),
        b_(slice(parameters_, input_size * output_size, output_size)) {
    normal_initialize(input_size, flatten(a_));
    b_ = 0;
  } // namespace xylo

  matrix forward(matrix_view input) override {
    matrix result = ::matmul_transposed(input, a_);

    for (std::size_t i = 0; i < input.num_rows(); ++i) {
      result[i] += b_;
    }
    return result;
  }

  matrix backward(matrix_view input, matrix_view backprop) override {
    return ::matmul(backprop, a_);
  }

  vector gradient(matrix_view input, matrix_view backprop) override {
    std::size_t input_size = input.num_cols();
    std::size_t output_size = backprop.num_cols();

    vector result({input_size * output_size + output_size});
    matrix_view d_a = fold<2>(slice(result, 0, input_size * output_size),
                              {output_size, input_size});
    vector_view d_b = slice(result, input_size * output_size, output_size);

    matmul(::transpose(backprop), input, d_a);
    d_b = 0;
    for (std::size_t i = 0; i < backprop.num_rows(); ++i) {
      d_b += backprop[i];
    }
    return result;
  }

  vector_view parameters() const override { return parameters_; }

protected:
  vector parameters_;
  matrix_view a_;
  vector_view b_;
};

using full_layer = matmul_layer;

#if 1
class convolution1d_1_layer : public layer {
public:
  convolution1d_1_layer(std::size_t input_channels, std::size_t output_channels,
                        std::string_view name = "")
      : layer(name),
        parameters_({output_channels * input_channels + output_channels}),
        a_(fold<2>(slice(parameters_, 0, output_channels * input_channels),
                   {output_channels, input_channels})),
        b_(slice(parameters_, output_channels * input_channels,
                 output_channels)) {
    he_initialize(input_channels, flatten(a_));
    b_ = 0;
  }

  matrix forward(matrix_view input) override {
    std::size_t input_channels = a_.num_cols();
    std::size_t output_channels = a_.num_rows();
    std::size_t num_batches = input.num_rows();
    std::size_t num_points = input.num_cols() / input_channels;

    matrix_view reshaped_input =
        fold<2>(input.flatten(), {num_batches * num_points, input_channels});

    matrix result({num_batches, num_points * output_channels});

    // Point per row, instead of batch per row. Output channel per col.
    matrix_view reshaped_result =
        fold<2>(flatten(result), {num_batches * num_points, output_channels});
    matmul_transposed(reshaped_input, a_, reshaped_result);

    for (std::size_t i = 0; i < num_batches * num_points; ++i) {
      reshaped_result[i] += b_;
    }
    return result;
  }

  matrix backward(matrix_view input, matrix_view backprop) override {
    std::size_t num_batches = input.num_rows();
    std::size_t output_channels = a_.num_rows();
    std::size_t num_points = backprop.num_cols() / output_channels;
    matrix_view reshaped_backprop =
        fold<2>(flatten(backprop), {num_batches * num_points, output_channels});

    std::size_t input_channels = a_.num_cols();
    matrix result({num_batches, num_points * input_channels});
    matrix_view reshaped_result =
        fold<2>(flatten(result), {num_batches * num_points, input_channels});
    matmul(reshaped_backprop, a_, reshaped_result);
    return result;
  }
  vector gradient(matrix_view input, matrix_view backprop) override {
    std::size_t num_batches = input.num_rows();
    std::size_t input_channels = a_.num_cols();
    std::size_t output_channels = a_.num_rows();
    std::size_t num_points = input.num_cols() / input_channels;

    matrix_view reshaped_input =
        fold<2>(flatten(input), {num_batches * num_points, input_channels});
    matrix_view reshaped_backprop =
        fold<2>(flatten(backprop), {num_batches * num_points, output_channels});

    vector result({output_channels * input_channels + output_channels});
    matrix_view d_a =
        fold<2>(slice(result, 0, output_channels * input_channels),
                {output_channels, input_channels});
    vector_view d_b =
        slice(result, output_channels * input_channels, output_channels);
    matmul(::transpose(reshaped_backprop), reshaped_input, d_a);
    d_b = 0;
    for (std::size_t i = 0; i < reshaped_backprop.num_rows(); ++i) {
      d_b += reshaped_backprop[i];
    }
    return result;
  }

  vector_view parameters() const override { return parameters_; }

private:
  vector parameters_;
  matrix_view a_;
  vector_view b_;
};
#else
class convolution1d_1_layer : public matmul_layer {
public:
  convolution1d_1_layer(std::size_t input_channels, std::size_t output_channels,
                        std::string_view name = "")
      : matmul_layer(input_channels, output_channels, name) {}

  matrix forward(matrix_view input) override {
    std::size_t input_channels = a_.num_cols();
    std::size_t output_channels = a_.num_rows();
    std::size_t num_batches = input.num_rows();
    std::size_t num_points = input.num_cols() / input_channels;

    matrix_view reshaped_input =
        fold<2>(input.flatten(), {num_batches * num_points, input_channels});

    matrix result({num_batches, num_points * output_channels});

    flatten(result) = flatten(matmul_layer::forward(reshaped_input));
    return result;
  }

  matrix backward(matrix_view input, matrix_view backprop) override {
    std::size_t num_batches = input.num_rows();
    std::size_t input_channels = a_.num_cols();
    std::size_t output_channels = a_.num_rows();
    std::size_t num_points = backprop.num_cols() / output_channels;
    matrix_view reshaped_backprop =
        fold<2>(flatten(backprop), {num_batches * num_points, output_channels});

    matrix result({num_batches, num_points * input_channels});

    // Input isn't actually used.
    flatten(result) = flatten(matmul_layer::backward(input, reshaped_backprop));
    return result;
  }

  vector gradient(matrix_view input, matrix_view backprop) override {
    std::size_t num_batches = input.num_rows();
    std::size_t input_channels = a_.num_cols();
    std::size_t output_channels = a_.num_rows();
    std::size_t num_points = input.num_cols() / input_channels;

    matrix_view reshaped_input =
        fold<2>(flatten(input), {num_batches * num_points, input_channels});
    matrix_view reshaped_backprop =
        fold<2>(flatten(backprop), {num_batches * num_points, output_channels});

    return matmul_layer::gradient(reshaped_input, reshaped_backprop);
  }

  vector_view parameters() const override { return matmul_layer::parameters(); }
};
#endif

// 3x3 convolution layer.
template <std::size_t signal_row, std::size_t signal_col>
class convolution2d_layer : public matmul_layer {
public:
  convolution2d_layer(std::size_t filter_size, std::size_t input_channels,
                      std::size_t output_channels, std::string_view name)
      // Output_channels is the number of filters. Length is of course the
      // length of the filter along the sequence, and input channels are the
      // innermost.
      : matmul_layer(filter_size * filter_size * input_channels,
                     output_channels, name),
        filter_size_(filter_size) {}

  matrix forward(matrix_view input) override {
    stretched_out_.emplace(im2col(input));
    matrix output = matmul_layer::forward(*stretched_out_);
    return matrix(fold<2>(
        flatten(output), {input.num_rows(), output.size() / input.num_rows()}));
  }

  vector gradient(matrix_view input, matrix_view backprop) override {
    matrix_view reshaped_backprop = matrix(
        fold<2>(flatten(backprop),
                {matrix_view(*stretched_out_).num_rows(),
                 backprop.size() / matrix_view(*stretched_out_).num_rows()}));
    return matmul_layer::gradient(*stretched_out_, reshaped_backprop);
  }
  matrix backward(matrix_view input, matrix_view loss) override {
    matrix stretched_out_flow = matmul_layer::backward(*stretched_out_, loss);
    return col2im(stretched_out_flow);
  }
  vector_view parameters() const override { return matmul_layer::parameters(); }

private:
  matrix im2col(matrix_view images) {
    std::size_t input_channels = images.num_cols() / signal_row / signal_col;
    matrix result({images.num_rows() * signal_row * signal_col,
                   filter_size_ * filter_size_ * input_channels});

    const std::size_t radius = filter_size_ / 2;
    std::size_t result_idx = 0;
    for (vector_view image : images) {
      tensor_view<3> image_matrix =
          fold<3>(image, {signal_row, signal_col, input_channels});
      for (std::size_t i = 0; i < signal_row; ++i) {
        std::size_t row_start = i - radius;
        for (std::size_t j = 0; j < signal_col; ++j) {
          std::size_t col_start = j - radius;

          vector_view result_row = result[result_idx++];
          tensor_view<3> result_block =
              fold<3>(result_row, {filter_size_, filter_size_, input_channels});
          for (std::size_t i = 0; i < filter_size_; ++i) {
            for (std::size_t j = 0; j < filter_size_; ++j) {
              std::size_t x = row_start + i;
              std::size_t y = col_start + j;
              bool out_of_bound =
                  x < 0 || y < 0 || x >= signal_row || y >= signal_col;
              // Don't try to use a ternary. Types won't match.
              if (out_of_bound) {
                result_block[i][j] = 0;
              } else {
#if 0
                lg() << "result block shape: "
                     << xeno::string::streamable(result_block.shape());
                lg() << "image matrix shape: "
                     << xeno::string::streamable(image_matrix.shape());
#endif
                result_block[i][j] = image_matrix[x][y];
              }
            }
          }
        }
      }
    }
    return result;
  }

  matrix col2im(matrix_view blocks) {
    std::size_t input_channels =
        blocks.num_cols() / filter_size_ / filter_size_;
    std::size_t num_images = blocks.num_rows() / signal_row / signal_col;
    matrix result({num_images, signal_row * signal_col * input_channels});

    return result;
  }

  std::optional<matrix> stretched_out_;
  std::size_t filter_size_;
};

class activation_layer : public layer {
public:
  explicit activation_layer(std::string_view name = "") : layer(name) {}
  vector gradient(matrix_view input, matrix_view backprop) override {
    return vector({0});
  }
  vector_view parameters() const override { return vector({0}); }
}; // namespace xylo

class relu_activation : public activation_layer {
public:
  explicit relu_activation(std::string_view name = "")
      : activation_layer(name) {}
  matrix forward(matrix_view input) override {
    matrix result({input.num_rows(), input.num_cols()});
    vector_view result_flattened = xylo::flatten(result);
    vector_view input_flattened = xylo::flatten(input);
    for (std::size_t i = 0; i < result.size(); ++i) {
      float val = input_flattened[i];
      result_flattened[i] = val > 0 ? val : 0;
    }
    return result;
  }
  matrix backward(matrix_view input, matrix_view backprop) override {
    matrix result({backprop.num_rows(), backprop.num_cols()});
    vector_view input_flattened = flatten(input);
    vector_view backprop_flattened = flatten(backprop);
    vector_view result_flattened = flatten(result);
    float num_dead = 0;
    for (std::size_t i = 0; i < result.size(); ++i) {
      result_flattened[i] = input_flattened[i] > 0 ? backprop_flattened[i] : 0;
      num_dead += input_flattened[i] > 0 ? 0 : 1;
    }
    // lg() << name_ << " dead fraction: " << num_dead / result.size();
    return result;
  }
};

class softmax_layer : public layer {
public:
  explicit softmax_layer(std::string_view name = "") : layer(name) {}
  matrix forward(matrix_view input) override {
    matrix result({input.num_rows(), input.num_cols()});
    matrix exponentials({input.num_rows(), input.num_cols()});
    exp(flatten(input), flatten(exponentials));

    for (std::size_t i = 0; i < input.num_rows(); ++i) {
      float sum = ::sum(exponentials[i]);
      result[i] = exponentials[i] / sum;
    }
    return result;
  }
  matrix backward(matrix_view input, matrix_view backprop) override {
    std::size_t batch_size = input.num_rows();
    std::size_t signal_length = input.num_cols();

    matrix result({batch_size, signal_length});
    matrix sigmas = forward(input);

    for (std::size_t i = 0; i < batch_size; ++i) {
      vector_view signal = sigmas[i];
      vector_view grads = backprop[i];
      matrix quadratic = ::matmul(fold<2>(signal, {signal_length, 1}),
                                  fold<2>(signal, {1, signal_length}));

      matrix linear{signal_length, signal_length};
      for (std::size_t j = 0; j < signal_length; ++j) {
        for (std::size_t k = 0; k < signal_length; ++k) {
          linear[j][k] = (j == k) ? signal[j] : 0.0f;
        }
      }
      matrix partial_diff = linear - quadratic;
      result[i] = flatten(
          ::matmul_transposed(partial_diff, grads.fold(1, signal_length)));
    }
    return result;
  }
  vector gradient(matrix_view input, matrix_view backprop) override {
    return vector({0});
  }
  vector_view parameters() const override { return vector({0}); }
};

class softmax_cross_entropy_layer : public softmax_layer {
public:
  explicit softmax_cross_entropy_layer(std::string_view name = "")
      : softmax_layer(name) {}
  matrix backward(matrix_view input, matrix_view backprop) override {
    return matrix(backprop);
  }
};

class matrix_var {
public:
  matrix_var() = default;
  matrix_var(const matrix &m) : holder_(std::make_unique<matrix>(m)) {}
  matrix_var(matrix &&m) : holder_(std::make_unique<matrix>(std::move(m))) {}

  void operator=(const matrix &m) { holder_ = std::make_unique<matrix>(m); }
  void operator=(matrix &&m) {
    holder_ = std::make_unique<matrix>(std::move(m));
  }

  const matrix &value() { return *holder_; }

private:
  std::unique_ptr<matrix> holder_;
};

class vector_var {
public:
  vector_var() = default;
  vector_var(const vector &v) : holder_(std::make_unique<vector>(v)) {}
  vector_var(vector &&v) : holder_(std::make_unique<vector>(std::move(v))) {}

  void operator=(const vector &v) { holder_ = std::make_unique<vector>(v); }
  void operator=(vector &&v) {
    holder_ = std::make_unique<vector>(std::move(v));
  }

  vector &value() { return *holder_; }

private:
  std::unique_ptr<vector> holder_;
};

class model {
public:
  void add_layer(std::unique_ptr<layer> &&l) {
    layers_.emplace_back(std::move(l));
  }

  matrix eval(matrix_view batch) const {
    matrix_var input = matrix(batch);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      input = layers_[i]->forward(input.value());
    }
    return input.value();
  }

  std::vector<matrix> forward(matrix_view batch) const {
    std::vector<matrix> input;
    input.emplace_back(batch);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      input.emplace_back(layers_[i]->forward(input[i]));
    }
    return input;
  }

  void set_parameters(vector_view parameters) {
    std::size_t curr_offset = 0;
    for (const auto &layer : layers_) {
      std::size_t layer_size = layer->parameters().size();
      layer->parameters() = slice(parameters, curr_offset, layer_size);
      curr_offset += layer_size;
    }
  }

  vector parameters() {
    vector result({parameter_size()});
    std::size_t curr_offset = 0;
    for (const auto &layer : layers_) {
      const std::size_t layer_size = layer->parameters().size();
      slice(result, curr_offset, layer_size) = layer->parameters();
      curr_offset += layer_size;
    }
    return result;
  }

  vector gradient(const std::vector<matrix> &input,
                  const matrix &target) const {
    vector result(parameter_size());
    matrix_var backprop = target;
    std::size_t curr_offset = result.size();

    for (std::size_t i = layers_.size() - 1; i > 0; --i) {
      const auto &layer = layers_[i];
      const std::size_t layer_size = layer->parameters().size();

      slice(result, curr_offset - layer_size, layer_size) =
          layer->gradient(input[i], backprop.value());
      backprop = layer->backward(input[i], backprop.value());
      curr_offset -= layer_size;
    }
    slice(result, 0, layers_[0]->parameters().size()) =
        layers_[0]->gradient(input[0], backprop.value());
    return result;
  }

  std::span<std::unique_ptr<layer>> layers() { return layers_; }

private:
  std::size_t parameter_size() const {
    std::size_t size = 0;
    for (const auto &layer : layers_) {
      size += layer->parameters().size();
    }
    return size;
  }

  std::vector<std::unique_ptr<layer>> layers_;
};

// output, external info per batch (e.g. label)
using loss_grad_func = std::function<matrix(matrix_view)>;
using loss_func = std::function<float(matrix_view)>;

matrix square_loss_grad(vector_view label, matrix_view output) {
  return output - fold<2>(label, {output.num_rows(), 1});
}

float square_loss(vector_view label, matrix_view output) {
  vector_view flattened_output = flatten(output);
  vector_view diff = flatten(output) - label;
  return dot(diff, diff) / label.size();
}

namespace {
template <typename T>
matrix convert_label_matrix(std::span<const T> labels,
                            std::size_t category_size) {
  matrix label_matrix({labels.size(), category_size});
  for (std::size_t i = 0; i < labels.size(); ++i) {
    vector_view v = label_matrix[i];
    v = 0;
    v[labels[i]] = 1.0;
  }
  return label_matrix;
}
} // namespace

// TODO: fix T at uint8_t. Categories larger than 255 will have some issues.
template <typename T>
matrix softmax_cross_entropy_loss_grad(std::span<const T> labels,
                                       std::size_t category_size,
                                       matrix_view output) {
  matrix result(output);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    result[i][labels[i]] -= 1;
  }
  return result;
}

matrix softmax_cross_entropy_loss_grad(matrix_view truth, matrix_view output) {
  return output - truth;
}

// class
class optimizer {
public:
  optimizer(model &m, float rate) : model_(m), rate_(rate) {}
  void set_rate(float rate) { rate_ = rate; }

  void step(matrix_view input, const loss_grad_func &loss_grad) {
    std::vector<matrix> inputs = model_.forward(input);

    matrix output = inputs.back();
    inputs.pop_back();

    matrix target = loss_grad(output);

    vector gradient = model_.gradient(inputs, target);
    vector parameters = next_parameters(model_.parameters(), gradient, rate_);
    model_.set_parameters(parameters);
  }

protected:
  virtual vector next_parameters(const vector &parameters,
                                 const vector &gradient, float rate) = 0;

private:
  float rate_;
  model &model_;
};

class sgd_optimizer : public optimizer {
public:
  sgd_optimizer(model &m, float rate, float weight_decay = 0.0f)
      : optimizer(m, rate), weight_decay_(weight_decay) {}

protected:
  vector next_parameters(const vector &parameters, const vector &gradient,
                         float rate) override {
    return parameters * (1 - weight_decay_) - gradient * rate;
  }

  float weight_decay_;
};

class momentum_optimizer : public optimizer {
public:
  momentum_optimizer(model &m, float rate)
      : optimizer(m, rate), velocity_{vector(0)} {}

protected:
  vector next_parameters(const vector &parameters, const vector &gradient,
                         float rate) override {
    if (velocity_->size() == 0) {
      velocity_.emplace(parameters.size());
      *velocity_ = 0;
    }

    vector_view velocity(*velocity_);

    // In-place version of v = rho * v + dx;
    velocity *= rho_;
    velocity += gradient;

    return parameters - velocity * rate;
  }

private:
  std::optional<vector> velocity_;
  const float rho_ = 0.9;
};

// Optimization video link:
// https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=3
class adam_optimizer : public optimizer {
public:
  adam_optimizer(model &m, float rate, float beta1 = 0.9, float beta2 = 0.999)
      : optimizer(m, rate), first_moment_(vector({0})),
        second_moment_(vector({0})), beta1_(beta1), beta2_(beta2) {}

protected:
  vector next_parameters(const vector &parameters, const vector &gradient,
                         float rate) override {
    if (first_moment_->size() == 0) {
      first_moment_.emplace(parameters.size());
      *first_moment_ = 0;
    }
    if (second_moment_->size() == 0) {
      second_moment_.emplace(parameters.size());
      *second_moment_ = 0;
    }

    vector_view first_moment(*first_moment_);
    vector_view second_moment(*second_moment_);

    first_moment = first_moment * beta1_ + gradient * (1 - beta1_);
    second_moment = second_moment * beta2_ + gradient * gradient * (1 - beta2_);

    vector first_unbias = first_moment / (1 - powf(beta1_, t));
    vector second_unbias = second_moment / (1 - powf(beta2_, t));

    t += 1;

    vector delta = first_unbias * rate / (::sqrt(second_unbias) + 1e-7);
    return parameters - first_unbias * rate / (::sqrt(second_unbias) + 1e-7);
  }

private:
  float t = 1;
  std::optional<vector> first_moment_;
  std::optional<vector> second_moment_;
  float beta1_;
  float beta2_;
};

} // namespace xylo

#endif // XYLO_NN_
