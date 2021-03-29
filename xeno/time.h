#ifndef XENO_TIME_
#define XENO_TIME_

#include <time.h>

#include <iomanip>
#include <sstream>
#include <string>

namespace xeno {
namespace time {

class duration {
public:
  constexpr duration() = default;

  std::strong_ordering operator<=>(const duration &other) const {
    if (auto cmp = time_.tv_sec <=> other.time_.tv_sec; cmp != 0) {
      return cmp;
    }
    return time_.tv_nsec <=> other.time_.tv_nsec;
  }
  bool operator==(const duration &other) const {
    return time_.tv_sec == other.time_.tv_sec &&
           time_.tv_nsec == other.time_.tv_nsec;
  }

  std::string debug_string() const {
    std::ostringstream os;
    if (time_.tv_sec != 0) {
      os << time_.tv_sec + time_.tv_nsec / 1'000'000'000.0 << "s";
      return os.str();
    } else if (time_.tv_nsec >= 1'000'000) {
      os << time_.tv_nsec / 1'000'000.0 << "ms";
    } else if (time_.tv_nsec >= 1'000) {
      os << time_.tv_nsec / 1'000.0 << "μs";
    } else {
      os << time_.tv_nsec << "ns";
    }
    return os.str();
  }

  ssize_t to_microseconds() {
    return time_.tv_sec * 1'000'000 + time_.tv_nsec / 1'000;
  }
  ssize_t to_nanoseconds() { return time_.tv_nsec; }

public:
  constexpr duration(timespec t) : time_(t) {}

  timespec time_;
};

class point {
public:
  constexpr point() = default;

  std::strong_ordering operator<=>(const point &other) const {
    if (auto cmp = time_.tv_sec <=> other.time_.tv_sec; cmp != 0) {
      return cmp;
    }
    return time_.tv_nsec <=> other.time_.tv_nsec;
  }
  bool operator==(const point &other) const {
    return time_.tv_sec == other.time_.tv_sec &&
           time_.tv_nsec == other.time_.tv_nsec;
  }

  std::string to_string() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);

    std::ostringstream oss;
    oss << std::put_time(&t, "%F %T") << '.' << std::left << std::setfill('0')
        << std::setw(6) << (time_.tv_nsec / 1'000);
    return oss.str();
  }

  // Deprecated.
  std::string to_localtime() const { return to_string(); }

  point start_of_second() const { return {{time_.tv_sec, 0}}; }
  point start_of_day() const {
    return {{time_.tv_sec - hour() * 3600 - minute() * 60 - second(), 0}};
  }

  int second() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_sec;
  }
  int minute() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_min;
  }
  int hour() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_hour;
  }
  int day_of_month() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_mday;
  }
  int month() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_mon + 1;
  }
  int year() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_year + 1900;
  }
  int day_of_week() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_wday;
  }
  int day_of_year() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_yday + 1;
  }
  bool is_dst() const {
    std::tm t;
    localtime_r(&time_.tv_sec, &t);
    return t.tm_isdst;
  }

public:
  constexpr point(timespec t) : time_(t) {}
  timespec time_;
};

namespace {
point now() {
  point p;
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  p.time_ = t;
  return p;
}

point mono_now() {
  point p;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  p.time_ = t;
  return p;
}
} // namespace

namespace {
constexpr timespec add(timespec t1, timespec t2) {
  timespec result;
  constexpr long giga = 1'000'000'000;

  // With potential regrouping
  result.tv_nsec = giga + t1.tv_nsec + t2.tv_nsec;
  result.tv_sec = -1 + (result.tv_nsec) / giga;

  result.tv_nsec %= giga;
  result.tv_sec += t1.tv_sec + t2.tv_sec;
  return result;
}

constexpr timespec neg(timespec t) { return {-t.tv_sec, -t.tv_nsec}; }
} // namespace

constexpr duration zero() { return duration({0, 0}); }
constexpr duration days(ssize_t n) { return duration({24 * 60 * 60 * n, 0}); }
constexpr duration hours(ssize_t n) { return duration({60 * 60 * n, 0}); }
constexpr duration minutes(ssize_t n) { return duration({60 * n, 0}); }
constexpr duration seconds(ssize_t n) { return duration({n, 0}); }
constexpr duration milliseconds(ssize_t n) {
  return duration({0, n * 1'000'000});
}
constexpr duration microseconds(ssize_t n) { return duration({0, n * 1'000}); }
constexpr duration nanoseconds(ssize_t n) { return duration({0, n}); }

constexpr point epoch() { return point({0, 0}); }
constexpr point seconds_since_epoch(ssize_t n) { return point({n, 0}); }
constexpr point milliseconds_since_epoch(ssize_t n) {
  return point({n / 1'000, (n % 1'000) * 1'000'000});
}
constexpr point microseconds_since_epoch(ssize_t n) {
  return point({n / 1'000'000, n % 1'000'000 * 1'000});
}

// TODO: implement this.
constexpr point from_iso8601() {
  // throw std::runtime_error("unimplemented");
  return epoch();
}

constexpr point operator-(point p) { return {neg(p.time_)}; }

constexpr duration operator-(point p1, point p2) {
  return add(p1.time_, neg(p2.time_));
}

constexpr point operator+(point p, duration d) { return add(p.time_, d.time_); }

constexpr duration operator+(duration d1, duration d2) {
  return add(d1.time_, d2.time_);
}

constexpr duration operator-(duration d1, duration d2) {
  return add(d1.time_, neg(d2.time_));
}

// Duration is not supposed to be negative.
constexpr duration operator*(uint32_t scale, duration d) {
  if (d.time_.tv_sec < 0 || d.time_.tv_nsec < 0)
    throw std::runtime_error("multiplying negative duration");

  int64_t product = d.time_.tv_nsec * scale;
  const int64_t giga = 1'000'000'000;

  return duration({scale * d.time_.tv_sec + product / giga, product % giga});
}

class stopwatch {
public:
  stopwatch(bool auto_start = true) {
    if (auto_start) {
      start();
    }
  }

  void start() { start_ = mono_now(); }

  duration read() { return mono_now() - start_; }

private:
  point start_ = epoch();
};

} // namespace time
} // namespace xeno

#endif // XENO_TIME_
