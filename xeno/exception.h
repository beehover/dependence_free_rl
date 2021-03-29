#ifndef XENO_EXCEPTION_
#define XENO_EXCEPTION_

#include <exception>
#include <experimental/source_location>
#include <functional>

#include <xeno/logging.h>

namespace xeno {

class error : public std::runtime_error {
public:
  error(std::string_view message,
        std::experimental::source_location location =
            std::experimental::source_location::current())
      : runtime_error(std::string(message)), location_(location) {}

  std::experimental::source_location location() const { return location_; }

private:
  std::experimental::source_location location_;
};

// We only retry xeno::error. If there are other types of exceptions thrown
// while this is called, the
template <typename Function, typename... Args>
void do_with_retry(Function &&f, Args &&... args) {
retry:
  try {
    return f(args...);
  } catch (const error &e) {
    log(log::error, e.location()) << e.what();
    goto retry;
  }
}

} // namespace xeno

#endif // XENO_EXCEPTION_
