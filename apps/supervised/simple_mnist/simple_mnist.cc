#include <xeno/logging.h>

#include <xylo/mnist.h>
#include <xylo/nn.h>
#include <xylo/tensor.h>

float calculate_accuracy(xylo::matrix_view batch,
                         std::span<uint8_t> label_batch) {
  float num_correct_predictions = 0;
  for (std::size_t i = 0; i < label_batch.size(); ++i) {
    xylo::vector_view v = batch[i];
    num_correct_predictions += v.argmax() == label_batch[i];
  }
  return num_correct_predictions / label_batch.size();
}

int main() {
  using conv_layer = xylo::convolution2d_layer<28, 28>;

  xylo::model model;
  // model.add_layer(std::make_unique<conv_layer>(3, 1, 8, "conv0"));
  model.add_layer(std::make_unique<xylo::full_layer>(784, 256, "full0"));
  model.add_layer(std::make_unique<xylo::relu_activation>("relu0"));
  model.add_layer(std::make_unique<xylo::full_layer>(256, 128, "full1"));
  model.add_layer(std::make_unique<xylo::relu_activation>("relu1"));
  model.add_layer(std::make_unique<xylo::full_layer>(128, 10, "full2"));
  model.add_layer(std::make_unique<xylo::softmax_cross_entropy_layer>(
      "softmax_cross_entropy"));

  xylo::sgd_optimizer opt(model, 1e-3, 1e-5);
  // xylo::mnist mnist("/home/xinli/git_repo/apps/supervised/simple_mnist");
  xylo::mnist mnist(".");

  xylo::matrix_view data = mnist.training_samples();
  std::span<uint8_t> labels = mnist.training_labels();
  constexpr int batch_size = 120;

  lg() << "start training";
  for (int epoch = 0;; ++epoch) {
    for (int i = 0; i < labels.size() / batch_size; ++i) {
      std::size_t start = i * batch_size;
      auto label_batch = labels.subspan(start, batch_size);
      auto loss_grad = std::bind_front(
          xylo::softmax_cross_entropy_loss_grad<uint8_t>, label_batch, 10);
      xylo::matrix_view slice = xylo::slice(data, start, batch_size);
      opt.step(slice, loss_grad);
    }
    // if (i % (60000 / batch_size) == 0 && i != 0) {
    float accuracy = calculate_accuracy(model.eval(mnist.testing_samples()),
                                        mnist.testing_labels());
    lg() << "accuracy " << epoch << ": " << accuracy;
    for (auto &layer : model.layers()) {
      lg() << "  layer " << layer->name();
      lg() << "  mean: " << mean(layer->parameters());
      lg() << "  variance: " << variance(layer->parameters());
      // lg() << "  raw data: " <<
      // xeno::string::streamable(layer->parameters());
    }
  }

  return 0;
}
