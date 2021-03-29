#ifndef XYLO_MNIST_H_
#define XYLO_MNIST_H_

#include <filesystem>
#include <span>
#include <string_view>

#include <xeno/endian.h>
#include <xeno/sys/file_descriptor.h>
#include <xeno/sys/io.h>

#include <xylo/tensor.h>

namespace xylo {

class mnist {
public:
  mnist(const std::filesystem::path &dir) {
    if (!std::filesystem::exists(dir / training_prep_filename) ||
        !std::filesystem::exists(dir / testing_prep_filename)) {
      preprocess(dir);
    }
    training_labels_mmap_ = load_label_file(dir / training_label_filename);
    testing_labels_mmap_ = load_label_file(dir / testing_label_filename);

    training_images_mmap_ = load_prep_file(dir / training_prep_filename);
    testing_images_mmap_ = load_prep_file(dir / testing_prep_filename);

    xylo::vector_view training_images_vec =
        xylo::borrow_vector(training_images_mmap_.span());
    xylo::vector_view testing_images_vec =
        xylo::borrow_vector(testing_images_mmap_.span());
    training_images_.emplace(
        reshap_images(dir / training_image_filename, training_images_vec));
    testing_images_.emplace(
        reshap_images(dir / testing_image_filename, testing_images_vec));
  }

  xylo::matrix_view training_samples() { return *training_images_; }
  xylo::matrix_view testing_samples() { return *testing_images_; }

  std::span<uint8_t> training_labels() {
    return training_labels_mmap_.span().subspan(label_header_size);
  }

  std::span<uint8_t> testing_labels() {
    return testing_labels_mmap_.span().subspan(label_header_size);
  }

private:
  constexpr static std::size_t label_header_size = 8;
  constexpr static char training_label_filename[] = "train-labels-idx1-ubyte";
  constexpr static char training_image_filename[] = "train-images-idx3-ubyte";
  constexpr static char training_prep_filename[] = "training.prep";
  constexpr static char testing_label_filename[] = "t10k-labels-idx1-ubyte";
  constexpr static char testing_image_filename[] = "t10k-images-idx3-ubyte";
  constexpr static char testing_prep_filename[] = "testing.prep";

  xylo::matrix_view reshap_images(const std::filesystem::path &image_path,
                                  xylo::vector_view vec) {
    auto mmap = xeno::sys::mmap<std::byte>(image_path, 16);
    auto num_images = xeno::from_wire<uint32_t>(mmap.span().subspan(4, 8));
    auto num_rows = xeno::from_wire<uint32_t>(mmap.span().subspan(8, 12));
    auto num_cols = xeno::from_wire<uint32_t>(mmap.span().subspan(12));
    return xylo::fold<2>(vec, {num_images, num_rows * num_cols});
  }

  xeno::sys::mmap<float>
  load_prep_file(const std::filesystem::path &prep_path) {
    return xeno::sys::mmap<float>(prep_path);
  }

  xeno::sys::mmap<uint8_t>
  load_label_file(const std::filesystem::path &label_path) {
    xeno::sys::mmap<std::byte> header(label_path, label_header_size);

    uint32_t magic_number =
        xeno::from_wire<uint32_t>(header.span().subspan(0, 4));
    if (magic_number != 2049) {
      throw xeno::error(
          xeno::string::strcat("magic number is not 2049: ", magic_number));
    }
    uint32_t size = xeno::from_wire<uint32_t>(header.span().subspan(4));

    auto result = xeno::sys::mmap<uint8_t>(label_path);
    if (size != result.span().size() - label_header_size) {
      throw xeno::error(xeno::string::strcat(
          "sizes don't match: header ", size, " vs. ", " actual ",
          result.span().size() - label_header_size));
    }
    return result;
  }

  void convert_image_file(const std::filesystem::path &image_path,
                          const std::filesystem::path &prep_path) {
    constexpr std::size_t image_header_size = 16;
    auto image_file = xeno::sys::mmap<std::byte>(image_path);
    auto header = image_file.span().subspan(0, image_header_size);

    uint32_t magic_number = xeno::from_wire<uint32_t>(header.subspan(0, 4));
    if (magic_number != 2051) {
      throw xeno::error(
          xeno::string::strcat("magic number is not 2051: ", magic_number));
    }

    uint32_t num_images = xeno::from_wire<uint32_t>(header.subspan(4, 4));
    uint32_t num_rows = xeno::from_wire<uint32_t>(header.subspan(8, 4));
    uint32_t num_cols = xeno::from_wire<uint32_t>(header.subspan(12));
    const uint64_t image_size = num_rows * num_cols;

    auto pixels = image_file.span().subspan(image_header_size);
    xeno::sys::file prep_file = xeno::sys::file::open_to_append(prep_path);
    xeno::sys::buffered_blocking_io<xeno::sys::file> io(prep_file);
    for (uint32_t i = 0; i < pixels.size(); ++i) {
      float pixel = static_cast<uint8_t>(pixels[i]) / 255.0;
      io.assured_write(xeno::to_native(pixel));
    }
  }

  void preprocess(const std::filesystem::path &dir) {
    convert_image_file(dir / training_image_filename,
                       dir / training_prep_filename);
    convert_image_file(dir / testing_image_filename,
                       dir / testing_prep_filename);
  }

  uint32_t num_images_;
  uint32_t num_rows_;
  uint32_t num_cols_;

  xeno::sys::mmap<uint8_t> training_labels_mmap_;
  xeno::sys::mmap<uint8_t> testing_labels_mmap_;

  xeno::sys::mmap<float> training_images_mmap_;
  xeno::sys::mmap<float> testing_images_mmap_;

  std::optional<xylo::matrix_view> training_images_;
  std::optional<xylo::matrix_view> testing_images_;
};
} // namespace xylo

#endif // XYLO_MNIST_H_
