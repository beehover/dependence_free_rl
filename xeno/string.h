#ifndef XENO_STRING_
#define XENO_STRING_

#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace xeno::string {

inline std::string_view strip_front(std::string_view sv) {
  auto first_pos = sv.find_first_not_of(" \t\r\n");
  if (first_pos == std::string_view::npos) {
    return "";
  }
  return sv.substr(first_pos);
}

inline std::string_view strip_back(std::string_view sv) {
  auto last_pos = sv.find_last_not_of(" \t\r\n");
  return sv.substr(0, last_pos + 1);
}

inline std::string_view strip(std::string_view sv) {
  auto first_pos = sv.find_first_not_of(" \t\r\n");

  if (first_pos == std::string_view::npos) {
    return "";
  }

  auto last_pos = sv.find_last_not_of(" \t\r\n");
  return sv.substr(first_pos, last_pos + 1 - first_pos);
}

template <typename... Types> std::string strcat(Types... args) {
  std::ostringstream oss;
  auto output = [&](const auto &arg) { oss << arg; };
  (..., output(args));
  return oss.str();
}

template <std::ranges::range T> std::string join(T &&v, char sep = ',') {
  std::string result;
  bool first = true;
  for (std::string_view s : v) {
    if (first) {
      first = false;
    } else {
      result += sep;
    }
    result += s;
  }
  return result;
}

inline std::vector<std::string_view> split(std::string_view s, char sep = ',') {
  std::vector<std::string_view> result;
  std::size_t last_end = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == sep) {
      result.push_back(s.substr(last_end, i - last_end));
      last_end = i + 1;
    }
  }
  result.push_back(s.substr(last_end));
  return result;
}

inline std::pair<std::string_view, std::string_view>
split_pair(std::string_view s, char sep = ',') {
  auto pos = s.find(sep);

  std::string_view first = s.substr(0, pos);

  std::string_view second =
      pos == std::string_view::npos ? "" : s.substr(pos + 1);
  return {first, second};
}

inline std::string streamable(const std::byte &b, std::string_view sep = "") {
  return std::to_string(static_cast<uint8_t>(b));
}

template <typename T> concept string_like = requires(T &t) {
  std::string_view(t);
};

#if 0
template <typename T> struct is_string_like : std::false_type {};
template <> struct is_string_like<std::string> : std::true_type {};
template <> struct is_string_like<std::string_view> : std::true_type {};
template <> struct is_string_like<char *> : std::true_type {};

template <typename T>
concept string_like = is_string_like<typename std::decay<T>::type>::value;
#endif

template <typename T1, typename T2>
std::string streamable(const std::pair<T1, T2> &p, std::string_view sep = ",") {
  std::ostringstream oss;
  oss << "(" << p.first << sep << p.second << ")";
  return oss.str();
}

template <typename T>
std::string streamable(const T &t, std::string_view sep = "") {
  std::ostringstream oss;
  oss << t;
  return oss.str();
}

template <typename T> concept range = requires(T &t) {
  t.begin();
  t.end();
};

template <typename T> concept non_string_range = range<T> && !string_like<T>;

template <non_string_range T>
std::string streamable(const T &r, std::string_view sep = ",") {
  std::ostringstream oss;
  oss << '[';
  bool first = true;
  for (auto &&item : r) {
    if (first) {
      first = false;
    } else {
      oss << sep;
    }
    oss << streamable(item, sep);
  }
  oss << ']';

  return oss.str();
}

} // namespace xeno::string

#endif // XENO_STRING_
