#ifndef XENO_LOGGING_
#define XENO_LOGGING_

#include <ctime>
#include <experimental/source_location>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include <xeno/sys/thread.h>
#include <xeno/time.h>

namespace std {
using source_location = experimental::source_location;
}

namespace xeno {
namespace logging {

extern thread_local std::string thread_name;

struct key {
  xeno::time::point timestamp;
  std::thread::id tid;
};

struct entry {
  std::source_location location;
  std::string content;
};

class log_store {
  log_store() = default;

private:
  std::map<key, entry> logs;
};

class logstream : public std::ostringstream {
public:
  enum level {
    info,
    warning,
    error,
    fatal,
  };

  logstream(level l = info, const std::source_location &location =
                                std::source_location::current())
      : level_(l), location_(location) {
    (*this) << std::boolalpha;
  }
  ~logstream() { output(); }

  logstream(logstream &&other) : std::ostringstream(std::move(other)) {}

protected:
  void output() {
    std::filesystem::path p(location_.file_name());

    std::ostringstream buffer;

    buffer << time::now().to_localtime() << ' ' << level() << ' ' << thread_name
           << '\t' << p.filename().string() << ":" << location_.line() << ":\t"
           << str() << std::endl;
    std::cerr << buffer.str();
  }

private:
  char level() {
    switch (level_) {
    case info:
      return 'I';
    case warning:
      return 'W';
    case error:
      return 'E';
    case fatal:
      return 'F';
    default:
      return ' ';
    }
  }

  enum level level_;
  std::experimental::source_location location_;
};

} // namespace logging
using log = logging::logstream;
} // namespace xeno

using lg = xeno::log;
#endif // XENO_LOGGING_
