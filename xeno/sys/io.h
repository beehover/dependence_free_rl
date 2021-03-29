#ifndef XENO_SYS_IO_
#define XENO_SYS_IO_

#include <span>
#include <stdexcept>
#include <vector>

#include <xeno/logging.h>

namespace xeno {
namespace sys {

template <typename T> class blocking_io {
public:
  blocking_io(T &conn) : connection_(conn) {}

  void assured_read(std::span<std::byte> buffer) {
    while (!buffer.empty()) {
      std::size_t num_read = connection_.read(buffer);
      if (num_read > buffer.size()) {
        throw std::runtime_error("read more than required");
      }
      buffer = buffer.subspan(num_read);
    }
  }

  void assured_read(std::size_t to_read, std::vector<std::byte> &buffer) {
    buffer.resize(to_read);
    assured_read(std::span(buffer.data(), to_read));
  }

  void assured_read(std::size_t to_read, std::string &buffer) {
    buffer.resize(to_read);
    assured_read(
        std::as_writable_bytes(std::span<char>(buffer.data(), to_read)));
  }

  void assured_read(std::string_view pattern, std::string &str) {
    throw std::runtime_error("not implemented");
  }

  void assured_write(std::span<const std::byte> buffer) {
    while (!buffer.empty()) {
      std::size_t num_written = connection_.write(buffer);
      if (num_written > buffer.size()) {
        throw std::runtime_error("wrote more than required");
      }
      buffer = buffer.subspan(num_written);
    }
  }

private:
  T &connection_;
};

template <typename T> class buffered_blocking_io {
public:
  buffered_blocking_io(T &conn) : connection_(conn) {}
  void assured_read(std::span<std::byte> s) {
    if (!buffer_.empty()) {
      if (buffer_.size() >= s.size()) {
        // Use data already in buffer_, and it'll suffice.
        std::copy_n(buffer_.begin(), s.size(), s.begin());

        // Copy excessive data back to buffer_.
        std::vector<std::byte> buffer = std::move(buffer_);
        buffer_.resize(buffer.size() - s.size());
        std::copy(buffer.begin() + s.size(), buffer.end(), buffer_.begin());
        return;
      } else {
        // Not enough data in the buffer. Copy them all and then possibly read
        // some more from the underlying stream.
        std::copy(buffer_.begin(), buffer_.end(), s.begin());
        s = s.subspan(buffer_.size());
      }
    }

    while (!s.empty()) {
      std::size_t num_read = connection_.read(s);

      if (num_read == 0) // EOF
        return;

      if (num_read > s.size()) {
        throw std::runtime_error("read more than required");
      }
      s = s.subspan(num_read);
    }
  }

  void assured_read(std::size_t to_read, std::vector<std::byte> &buffer) {
    buffer.resize(to_read);
    assured_read(std::span(buffer.data(), to_read));
  }

  void assured_read(std::size_t to_read, std::string &buffer) {
    buffer.resize(to_read);
    assured_read(
        std::as_writable_bytes(std::span<char>(buffer.data(), to_read)));
  }

  void assured_read(std::string_view pattern, std::string &str) {
    std::size_t capacity;
    std::size_t real_size = 0;

    if (!buffer_.empty()) {
      // If we have already got stuff left in our buffer, then start from that
      // stuff. Setting capacity to real size, so that when we try, we'll skip
      // the read and immediately increase the capacity size.
      copy_from_buffer(str);
      capacity = real_size = str.size();
    } else {
      // Starting with an empty buffer, we'll set capacity to a pre-defined
      // value.
      constexpr std::size_t init_capacity = 128;
      str.clear();
      capacity = init_capacity;
    }

  retry:
    if (str.size() < capacity) {
      str.resize(capacity);
    }

    std::span<char> read_span(str.data() + real_size, capacity - real_size);

    std::size_t num_read = connection_.read(std::as_writable_bytes(read_span));

    real_size += num_read;

    std::string_view sv(str.data(), real_size);
    std::size_t pos = sv.find(pattern);
    if (pos != std::string_view::npos) {
      // Found it. Rewind bytes past the pattern.
      goto success;
    }

    // Haven't found it, but we're not reading anything from the connection,
    // even if we meant to read.  Must be EOF. Bail.
    if (capacity - real_size != 0 && num_read == 0) {
      str.resize(real_size);
      return;
    }

    if (real_size == capacity) {
      capacity *= 2;
    }
    goto retry;

  success:
    std::size_t num_rewind_bytes = real_size - (pos + pattern.size());
    real_size -= num_rewind_bytes;
    std::span<const char> rewind_span =
        std::span<const char>(str.data() + real_size, num_rewind_bytes);
    copy_to_buffer(std::as_bytes(rewind_span));
    str.resize(real_size);
    str.shrink_to_fit();
  }

  void assured_write(std::span<const std::byte> buffer) {
    while (!buffer.empty()) {
      std::size_t num_written = connection_.write(buffer);
      if (num_written > buffer.size()) {
        num_written = buffer.size();
      }
      buffer = buffer.subspan(num_written);
    }
  }

  void assured_write(std::string_view sv) {
    std::span<const char> s(sv.begin(), sv.end());
    assured_write(std::as_bytes(s));
  }

private:
  void copy_to_buffer(std::span<const std::byte> s) {
    buffer_.resize(s.size());
    std::copy(s.begin(), s.end(), buffer_.begin());
  }
  void copy_from_buffer(std::span<std::byte> s) {
    std::copy(buffer_.begin(), buffer_.end(), s.begin());
  }
  void copy_from_buffer(std::string &s) {
    s.resize(buffer_.size());
    std::span<char> copy_span(s.begin(), s.end());
    copy_from_buffer(std::as_writable_bytes(copy_span));
  }

  std::size_t read(std::size_t size, std::byte *ptr) {
    return connection_.read(std::span<std::byte>(ptr, size));
  }

  std::size_t prefetch(std::size_t size_diff) {
    std::size_t old_size = buffer_.size();
    std::size_t new_size = size_diff + old_size;

    buffer_.resize(new_size);
    std::size_t size_read = read(size_diff, buffer_.data() + old_size);

    buffer_.resize(old_size + size_read);
    return buffer_.size();
  }

  std::size_t peek(std::span<std::byte> s) {
    if (s.size() > buffer_.size()) {
      prefetch(s.size() - buffer_.size());
      std::copy(buffer_.begin(), buffer_.end(), s.begin());
      return buffer_.size();
    } else {
      std::copy(buffer_.begin(), buffer_.end(), s.begin());
      return buffer_.size();
    }
  }

  T &connection_;
  std::vector<std::byte> buffer_;
};

} // namespace sys
} // namespace xeno

#endif // XENO_SYS_IO_
