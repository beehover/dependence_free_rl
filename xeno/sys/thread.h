#ifndef XENO_SYS_THREAD_
#define XENO_SYS_THREAD_

#include <signal.h>

#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>

namespace xeno {
namespace sys {

class thread {
public:
  explicit thread(std::string_view name = "") : name_(name), pid_(0) {
    pthread_attr_init(&attr_);
  }

  thread(const thread &) = delete;
  ~thread() { pthread_attr_destroy(&attr_); }

  void operator=(const thread &) = delete;

  pthread_t native_handle() { return pthread_self(); }

  template <typename F, typename... Args> void run(F &&f, Args &&... args) {
    if (joinable()) {
      throw std::runtime_error("launching on joinable thread");
    }
    closure_ = std::bind(f, args...);
    pthread_create(&pid_, &attr_, execute_closure, this);
  }

  void set_name(std::string_view s) { name_ = s; }
  std::string_view get_name() { return name_; }

  bool joinable() { return pid_ != 0; }
  void join();

  void cancel() { pthread_cancel(pid_); }

private:
  static void *execute_closure(void *p);

  std::string name_;
  pthread_t pid_;
  pthread_attr_t attr_;
  std::function<void()> closure_;
};

class thread_pool {
public:
  thread_pool(std::size_t num_threads) {}
};

} // namespace sys
} // namespace xeno

#endif // XENO_SYS_THREAD_
