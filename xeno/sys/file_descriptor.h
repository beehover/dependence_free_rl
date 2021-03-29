#ifndef XENO_SYS_DESCRIPTOR_
#define XENO_SYS_DESCRIPTOR_

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include <xeno/exception.h>
#include <xeno/string.h>

namespace xeno {
namespace sys {
class descriptor {
public:
  ~descriptor() {
    if (handle_ >= 0) {
      close(handle_);
    }
  }
  using handle_t = int;

  constexpr descriptor() = default;
  descriptor(descriptor &&other) : descriptor(other.handle_) {
    handle_ = other.handle_;
    other.handle_ = -1;
  }

  // Can't copy. Must move. Otherwise the underlying C descriptor will be closed
  // multiple times.
  descriptor(const descriptor &other) = delete;

  descriptor &operator=(const descriptor &other) = delete;
  descriptor &operator=(descriptor &&other) {
    handle_ = other.handle_;
    other.handle_ = -1;
    return *this;
  }
  handle_t get_handle() { return handle_; }

private:
  static constexpr bool error_num_relevant(int err_num, bool is_write) {
    std::array<int, 5> read_relevant_errors{EBADF, EFAULT, EINVAL, EIO, EISDIR};
    std::array<int, 10> write_relevant_errors{
        EBADF,  EDESTADDRREQ, EDQUOT, EFAULT, EFBIG,
        EINVAL, EIO,          ENOSPC, EPERM,  EPIPE};
    if (is_write) {
      return std::ranges::find(write_relevant_errors, err_num) !=
             write_relevant_errors.end();
    } else {
      return std::ranges::find(read_relevant_errors, err_num) !=
             read_relevant_errors.end();
    }
  }

  static std::size_t
  progress(std::size_t bytes_done, int err_num, bool is_write = false,
           std::experimental::source_location loc =
               std::experimental::source_location::current()) {
    if (bytes_done > 0)
      return bytes_done;

    if (bytes_done < 0)
      throw xeno::error(strerror(err_num), loc);

    // bytes_done == 0 from now on.
    if (error_num_relevant(err_num, is_write))
      throw xeno::error(strerror(err_num), loc);

    return 0;
  }

public:
  std::size_t read(std::span<char> buffer) {
    int bytes_done = ::read(handle_, buffer.data(), buffer.size());
    return progress(bytes_done, errno, false);
  }
  std::size_t write(std::span<const char> buffer) {
    int bytes_done = ::write(handle_, buffer.data(), buffer.size());
    return progress(bytes_done, errno, true);
  }
  std::size_t read(std::span<std::byte> buffer) {
    int bytes_done = ::read(handle_, buffer.data(), buffer.size());
    return progress(bytes_done, errno, false);
  }
  std::size_t write(std::span<const std::byte> buffer) {
    int bytes_done = ::write(handle_, buffer.data(), buffer.size());
    return progress(bytes_done, errno, true);
  }

  constexpr handle_t get_handle() const noexcept { return handle_; }

protected:
  constexpr descriptor(handle_t handle) : handle_(handle) {}
  handle_t handle_ = -1;
};

class address {
public:
  enum class family_enum {
    unix = AF_UNIX,
    local = AF_UNIX,
    ipv4 = AF_INET,
    ipv6 = AF_INET6,
  };

  enum class type_enum {
    tcp = SOCK_STREAM,
    udp = SOCK_DGRAM,
    raw = SOCK_RAW,
  };

  constexpr address(sockaddr *addr, std::size_t len) : addr_(addr), len_(len) {}

  constexpr sockaddr *addr() const { return addr_; }
  constexpr std::size_t len() const { return len_; }

  constexpr family_enum family() const {
    return static_cast<family_enum>(addr_->sa_family);
  }

private:
  sockaddr *addr_;
  std::size_t len_;
};

// Not thread safe.
address local_tcp_address(in_port_t port) {
  static struct sockaddr_in6 native_addr = {
      .sin6_family = AF_INET6,
      .sin6_port = port,
      .sin6_addr = in6addr_any,
  };
  return address(reinterpret_cast<sockaddr *>(&native_addr),
                 sizeof(native_addr));
}

class resolver {
public:
  resolver(std::string_view host,
           address::type_enum t = address::type_enum::tcp) {
    std::string host_str(host);
    getaddrinfo(host_str.c_str(), nullptr, nullptr, &result_);

    for (auto *ptr = result_; ptr; ptr = ptr->ai_next) {
      if (static_cast<address::type_enum>(ptr->ai_socktype) != t)
        continue;
      addrs_.emplace_back(ptr->ai_addr, ptr->ai_addrlen);
    }
  }
  ~resolver() { freeaddrinfo(result_); }
  std::vector<address> addrs() const noexcept { return addrs_; }

private:
  std::vector<address> addrs_;
  addrinfo *result_ = nullptr;
};

class socket : public descriptor {
public:
  static socket open(std::string_view host, int32_t port = 80,
                     address::type_enum t = address::type_enum::tcp) {
    resolver r(host, t);

    if (r.addrs().empty()) {
      throw xeno::error(
          xeno::string::strcat("can't resolve domain name ", host));
    }

    for (auto &addr : r.addrs()) {
      socket s(addr.family(), t);
      reinterpret_cast<sockaddr_in *>(addr.addr())->sin_port = htons(port);
      if (s.connect(addr)) {
        return s;
      }
    }
    throw xeno::error(
        xeno::string::strcat("can't connect to ", host, ":", port));
  }

  static socket create(int32_t port = 80,
                       address::type_enum t = address::type_enum::tcp,
                       int backlog = 1024) {
    address addr = local_tcp_address(port);
    socket s(addr.family(), t);
    reinterpret_cast<sockaddr_in *>(addr.addr())->sin_port = htons(port);
    s.bind(addr);
    s.listen(backlog);
    return s;
  }

  void bind(address addr) {
    if (::bind(handle_, addr.addr(), addr.len()) == -1) {
      throw xeno::error(xeno::string::strcat("binding failed"));
    };
  }
  void listen(int backlog) {
    if (::listen(handle_, backlog) == -1) {
      throw xeno::error(xeno::string::strcat("listening failed"));
    };
  }
  socket accept() { return ::accept(handle_, nullptr, nullptr); }

  constexpr socket() = default;
  socket(socket &&other) = default;

  socket &operator=(socket &&other) = default;

  bool connect(address addr) {
    return ::connect(handle_, addr.addr(), addr.len()) != -1;
  }

private:
  socket(address::family_enum af, address::type_enum t)
      : descriptor(::socket(static_cast<int>(af), static_cast<int>(t), 0)) {}
  constexpr socket(handle_t handle) : descriptor(handle){};
};

class string_view_mmap;

class file : public descriptor {
public:
  static void create(const std::filesystem::path &p) {
    open(p, O_WRONLY | O_CREAT);
  }
  static file open_to_read(const std::filesystem::path &p) {
    return open(p, O_RDONLY);
  }
  static file open_to_append(const std::filesystem::path &p) {
    return open(p, O_WRONLY | O_CREAT | O_APPEND);
  }
  static file open_to_mmap(const std::filesystem::path &p) {
    return open(p, O_RDWR | O_CREAT);
  }
  // Implementation after string_view_mmap;
  static std::string open_as_string(const std::filesystem::path &p);

  constexpr file() = default;
  file(file &&other) = default;

  file &operator=(const file &other) = delete;

protected:
  static file open(const std::filesystem::path &p, int flags) {
    return file(::open(p.string().c_str(), flags, 0660));
  }

  constexpr file(handle_t handle) : descriptor(handle) {
    if (handle == -1) {
      throw xeno::error(strerror(errno));
    }
  }
};

template <typename T> class mmap {
public:
  mmap() = default;

  // Size and pos are in terms of mapped objects.
  mmap(const std::filesystem::path &p, std::size_t size = -1) {
    file f = file::open_to_mmap(p);
    if (size == -1) {
      size = file_size(p) / sizeof(T);
    }

    if (size == 0)
      return;

    T *ptr = reinterpret_cast<T *>(::mmap(nullptr, size * sizeof(T),
                                          PROT_READ | PROT_WRITE, MAP_SHARED,
                                          f.get_handle(), 0));
    if (ptr == reinterpret_cast<T *>(-1))
      throw xeno::error("mmap failed");

    data_ = std::span(ptr, size);
  }

  ~mmap() {
    if (data_.data() == nullptr)
      return;

    if (::munmap(data_.data(), data_.size() * sizeof(T)) == -1)
      lg(lg::error) << "munmap failed";
  }

  mmap(const mmap<T> &) = delete;
  mmap(mmap<T> &&other) {
    data_ = other.data_;
    other.data_ = std::span<T>{};
  }

  void operator=(const mmap<T> &) = delete;
  void operator=(mmap<T> &&other) {
    data_ = other.data_;
    other.data_ = std::span<T>{};
  }

  std::span<T> span() const { return data_; }

private:
  std::span<T> data_;
};

class string_view_mmap {
public:
  string_view_mmap() = default;

  string_view_mmap(const std::filesystem::path &p, std::size_t size = -1) {
    file f = file::open_to_read(p);
    if (size == -1) {
      size = file_size(p);
    }

    if (size == 0)
      return;

    const char *ptr = reinterpret_cast<const char *>(
        ::mmap(nullptr, size, PROT_READ, MAP_SHARED, f.get_handle(), 0));
    if (ptr == reinterpret_cast<char *>(-1))
      throw xeno::error("mmap failed");

    data_ = std::string_view(ptr, size);
  }

  ~string_view_mmap() {
    if (data_.data() == nullptr)
      return;

    if (::munmap(const_cast<char *>(data_.data()), data_.size()) == -1)
      lg(lg::error) << "munmap failed";
  }

  string_view_mmap(const string_view_mmap &) = delete;

  string_view_mmap(string_view_mmap &&other) {
    data_ = other.data_;
    other.data_ = std::string_view{};
  }

  void operator=(const string_view_mmap &) = delete;
  void operator=(string_view_mmap &&other) {
    data_ = other.data_;
    other.data_ = std::string_view{};
  }

  std::string_view string_view() {
    return std::string_view(data_.data(), data_.size());
  }

private:
  std::string_view data_;
};

std::string file::open_as_string(const std::filesystem::path &p) {
  // TODO: use mmap.
  string_view_mmap mmap(p);
  return std::string(mmap.string_view());
}

} // namespace sys
} // namespace xeno

#endif // XENO_SYS_DESCRIPTOR_
