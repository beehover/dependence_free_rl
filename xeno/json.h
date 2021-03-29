#ifndef XENO_JSON_
#define XENO_JSON_

#include <algorithm>
#include <functional>
#include <map>
#include <ranges>
#include <string_view>
#include <variant>
#include <vector>

#include <xeno/logging.h>
#include <xeno/string.h>

namespace xeno {
namespace json {

class element {
public:
  element() = default;

  // TODO: Turn float into double. Also use std::from_chars to replace the
  // number parsing.
  using val_t =
      std::variant<std::monostate, bool, float, int64_t, std::string,
                   std::vector<element>, std::map<std::string, element>>;
  constexpr void set_bool(bool b) noexcept { value_ = b; }
  constexpr void set_float(float num) noexcept { value_ = num; }
  constexpr void set_number(float num) noexcept { value_ = num; }
  constexpr void set_integer(int64_t num) noexcept { value_ = num; }
  constexpr void set_string(std::string_view str) noexcept {
    value_ = std::string(str);
  }
  constexpr void set_array(std::vector<element> &&v) { value_ = std::move(v); }
  constexpr void set_object(std::map<std::string, element> &&m) {
    value_ = std::move(m);
  }
  constexpr void set_null() noexcept { value_ = std::monostate(); }

  constexpr bool has_bool() const noexcept {
    return std::holds_alternative<bool>(value_);
  }
  constexpr bool has_number() const noexcept {
    return has_integer() || has_float();
  }
  constexpr bool has_integer() const noexcept {
    return std::holds_alternative<int64_t>(value_);
  }
  constexpr bool has_float() const noexcept {
    return std::holds_alternative<float>(value_);
  }
  constexpr bool has_string() const noexcept {
    return std::holds_alternative<std::string>(value_);
  }
  constexpr bool has_array() const noexcept {
    return std::holds_alternative<std::vector<element>>(value_);
  }
  constexpr bool has_object() const noexcept {
    return std::holds_alternative<std::map<std::string, element>>(value_);
  }
  constexpr bool has_null() const noexcept {
    return std::holds_alternative<std::monostate>(value_);
  }

  constexpr bool get_bool() const noexcept { return std::get<bool>(value_); }
  constexpr float get_float() const noexcept { return std::get<float>(value_); }
  constexpr int64_t get_integer() const noexcept {
    return std::get<int64_t>(value_);
  }
  constexpr float get_number() const noexcept {
    if (has_float())
      return std::get<float>(value_);
    else
      return std::get<int64_t>(value_);
  }
  constexpr std::string_view get_string() const noexcept {
    return std::get<std::string>(value_);
  }
  constexpr const std::vector<element> &get_array() const noexcept {
    return std::get<std::vector<element>>(value_);
  }
  constexpr std::vector<element> &get_mutable_array() noexcept {
    if (!has_array()) {
      value_.emplace<std::vector<element>>();
    }
    return std::get<std::vector<element>>(value_);
  }
  constexpr const std::map<std::string, element> &get_object() const noexcept {
    return std::get<std::map<std::string, element>>(value_);
  }
  constexpr std::map<std::string, element> &get_mutable_object() noexcept {
    if (!has_object()) {
      value_.emplace<std::map<std::string, element>>();
    }
    return std::get<std::map<std::string, element>>(value_);
  }

  bool operator==(bool b) {
    if (!has_bool())
      return false;
    return b == get_bool();
  }
  bool operator==(float f) {
    if (!has_number())
      return false;
    return f == get_number();
  }
  bool operator==(int64_t f) {
    if (!has_integer())
      return false;
    return f == get_integer();
  }
  bool operator==(std::string_view s) {
    if (!has_string())
      return false;
    return s == get_string();
  }

  void operator=(bool b) { set_bool(b); }
  void operator=(float f) { set_float(f); }
  void operator=(int64_t i) { set_integer(i); }
  void operator=(std::string_view s) { set_string(s); }

  element &operator[](std::string_view str) {
    if (!has_object())
      set_object(std::map<std::string, element>());
    return get_mutable_object()[std::string(str)];
  }
  const element &operator[](std::string_view str) const {
    const auto it = get_object().find(std::string(str));
    const auto &[_, val] = *it;
    return val;
  }

  element &operator[](std::size_t index) {
    if (!has_array())
      set_array(std::vector<element>());
    return get_mutable_array()[index];
  }

  const element &operator[](std::size_t index) const {
    return get_array()[index];
  }

  std::string to_string() const {
    std::ostringstream oss;
    oss << std::boolalpha;
    to_string_helper(oss);
    return oss.str();
  }

private:
  void to_string_helper(std::ostringstream &os) const {
    if (has_bool()) {
      os << get_bool();
      return;
    }
    if (has_integer()) {
      os << get_integer();
      return;
    }
    if (has_number()) {
      os << get_number();
      return;
    }
    if (has_string()) {
      os << std::quoted(get_string());
      return;
    }
    if (has_object()) {
      os << '{';
      bool is_first = true;
      for (const auto &[key, value] : get_object()) {
        if (is_first) {
          is_first = false;
        } else {
          os << ',';
        }
        os << std::quoted(key) << ':' << value.to_string();
      }
      os << '}';
      return;
    }
    if (has_array()) {
      os << '[';
      bool is_first = true;
      for (const auto &value : get_array()) {
        if (is_first) {
          is_first = false;
        } else {
          os << ',';
        }
        os << value.to_string();
      }
      os << ']';
      return;
    }
    if (has_null()) {
      os << "null";
      return;
    }
  }

  val_t value_;
};

class data_builder {
public:
  virtual void on_start_object() = 0;
  virtual void on_end_object() = 0;
  virtual void on_start_array() = 0;
  virtual void on_end_array() = 0;
  virtual void on_key() = 0;
  virtual void on_element() = 0;
  virtual void on_string(std::string_view array) = 0;
  virtual void on_number(std::string_view null) = 0;
  virtual void on_literal(std::string_view literal) = 0;
};

class null_builder : public data_builder {
public:
  void on_start_object() override {}
  void on_end_object() override {}
  void on_start_array() override {}
  void on_end_array() override {}
  void on_key() override {}
  void on_element() override {}
  void on_string(std::string_view str) override {}
  void on_number(std::string_view num) override {}
  void on_literal(std::string_view literal) override {}
};

class default_builder : public data_builder {
public:
  explicit default_builder(element &e) : element_(e) {}

  void on_start_object() override { chain_.back().get().get_mutable_object(); }
  void on_end_object() override { chain_.pop_back(); }
  void on_start_array() override { chain_.back().get().get_mutable_array(); }
  void on_end_array() override { chain_.pop_back(); }

  void on_key() override { state_ = state::ready_for_key; }

  void on_element() override {
    state_ = state::ready_for_val;

    if (chain_.empty()) {
      chain_.emplace_back(element_);
      return;
    }

    element &curr = chain_.back();

    if (curr.has_array()) {
      chain_.emplace_back(curr.get_mutable_array().emplace_back());
    } else if (curr.has_object()) {
      chain_.emplace_back(curr.get_mutable_object()[std::string(key_)]);
    } else {
      chain_.emplace_back(element_);
    }
  }

  void on_string(std::string_view str) override {
    std::stringstream ss;
    ss.str(std::string{str});

    std::string unquoted;
    ss >> std::quoted(unquoted);

    if (state_ == state::ready_for_key) {
      key_ = std::move(unquoted);
    } else {
      chain_.back().get().set_string(std::move(unquoted));
      chain_.pop_back();
    }
  }

  void on_number(std::string_view num) override {
    const auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    if (std::ranges::all_of(num, is_digit)) {
      chain_.back().get().set_integer(std::stoll(std::string(num)));
    } else {
      chain_.back().get().set_number(std::stof(std::string(num)));
    }
    chain_.pop_back();
  }
  void on_literal(std::string_view literal) override {
    if (literal == "true") {
      chain_.back().get().set_bool(true);
    } else if (literal == "false") {
      chain_.back().get().set_bool(false);
    }
    chain_.pop_back();
  }

private:
  enum class state {
    init,
    ready_for_val,
    ready_for_key,
  };

  element &element_;
  std::vector<std::reference_wrapper<element>> chain_;
  state state_ = state::ready_for_val;
  std::string key_;
};

namespace {
null_builder _null_builder;
}

class parser {
public:
  class failure : public std::runtime_error {
  public:
    failure() : std::runtime_error("") {}
    failure(std::string_view what_arg)
        : std::runtime_error(std::string(what_arg)) {}
  };

  parser(std::string_view content, data_builder &builder = _null_builder)
      : curr_(0), content_(content), builder_(builder) {
    consume_json();
  }

private:
  struct segment {
    std::size_t pos = 0;
    std::size_t size = 0;

    std::string debug_string() {
      return std::string("pos ") + std::to_string(pos) + " size " +
             std::to_string(size);
    }

    void operator+=(segment other) { size += other.size; }
  };

  static constexpr bool is_sign(char c) { return c == '+' || c == '-'; }
  static constexpr bool is_none_zero_digit(char c) {
    return c > '0' && c <= '9';
  }
  static constexpr bool is_digit(char c) { return c >= '0' && c <= '9'; }
  static constexpr bool is_hex_digit(char c) {
    return is_digit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F';
  }
  static constexpr bool is_ws(char c) {
    std::array<char, 4> spaces{0x20, 0xa, 0xd, 0x9};
    const auto equal_to_c = [c](char space) -> bool { return space == c; };
    return std::any_of(spaces.begin(), spaces.end(), equal_to_c);
  }

  segment consume_symbol(char c) noexcept(false) {
    // LOG() << "looking for symbol " << c << "@" << curr_;
    // LOG() << "got symbol " << peek();
    if (peek() != c) {
      std::ostringstream oss;
      oss << "Looking for character " << c << " but got " << peek();
      throw failure(oss.str());
    }
    return consume_char();
  }

  segment consume_literal(std::string_view literal) {
    segment result = curr_zero_segment();
    for (int i = 0; i < literal.size(); ++i) {
      if (peek() != literal[i]) {
        throw failure(xeno::string::strcat("Looking for value ", literal));
      }
      result += consume_char();
    }

    builder_.on_literal(literal);

    return result;
  }

  segment consume_json() noexcept(false) { return consume_element(); }

  segment consume_value() noexcept(false) {
    char c = peek();
    switch (c) {
    case 't':
      return consume_literal("true");
    case 'f':
      return consume_literal("false");
    case 'n':
      return consume_literal("null");
    case '{':
      return consume_object();
    case '[':
      return consume_array();
    case '"':
      return consume_string();
    default:
      if (is_sign(c) || is_digit(c))
        return consume_number();

      std::ostringstream oss;
      oss << "Looking for object, array, string, number, true, false or null";
      throw failure(oss.str());
    }
  }

  segment consume_object() noexcept(false) {
    segment result = consume_symbol('{');
    builder_.on_start_object();

    // Get past the whitespaces to see which route we're going.
    segment ws_segment = consume_ws();
    bool has_member = (peek() != '}');
    restore(ws_segment.pos);

    result += has_member ? consume_members() : consume_ws();

    result += consume_symbol('}');
    builder_.on_end_object();
    return result;
  }

  segment consume_members() noexcept(false) {
    segment result = consume_member();

    if (peek() == ',') {
      result += consume_symbol(',');
      result += consume_members();
    }

    return result;
  }

  segment consume_member() noexcept(false) {
    builder_.on_key();

    segment result = consume_ws();
    result += consume_string();
    result += consume_ws();

    result += consume_symbol(':');

    result += consume_element();
    return result;
  }

  segment consume_array() noexcept(false) {
    segment result = consume_symbol('[');
    builder_.on_start_array();

    // Get past the whitespaces to see which route we're going.
    segment ws_segment = consume_ws();
    bool has_element = (peek() != ']');
    restore(ws_segment.pos);

    result += has_element ? consume_elements() : consume_ws();
    result += consume_symbol(']');
    builder_.on_end_array();
    return result;
  }

  segment consume_elements() noexcept(false) {
    segment result = consume_element();

    if (peek() == ',') {
      result += consume_symbol(',');
      result += consume_elements();
    }

    return result;
  }

  segment consume_element() noexcept(false) {
    builder_.on_element();
    segment result = consume_ws();
    result += consume_value();
    result += consume_ws();
    return result;
  }

  std::string_view segment_str(segment seg) {
    return content_.substr(seg.pos, seg.size);
  }

  segment consume_string() noexcept(false) {
    segment result = consume_symbol('"');
    result += consume_characters();
    result += consume_symbol('"');

    builder_.on_string(segment_str(result));
    return result;
  }

  segment consume_characters() noexcept {
    segment result = curr_zero_segment();

    for (;;) {
      switch (peek()) {
      case '"':
      case '\0':
        return result;

      case '\\':
        result += consume_char();
        result += consume_escape();

      default:
        result += consume_char();
      }
    }
  }

  segment consume_escape() noexcept(false) {
    segment result = curr_zero_segment();

    switch (peek()) {
    case '"':
    case '\\':
    case '/':
    case 'b':
    case 'f':
    case 'n':
    case 'r':
    case 't':
      result += consume_char();
      return result;
    case 'u':
      result += consume_char();
      for (int i = 0; i < 4; ++i) {
        if (!is_hex_digit(peek())) {
          throw failure();
        }
        result += consume_char();
      }
      return result;
    default:
      throw failure();
    }
  }

  segment consume_number() noexcept(false) {
    segment result = consume_integer();
    result += consume_fraction();
    result += consume_exponent();

    builder_.on_number(segment_str(result));

    return result;
  }

  segment consume_integer() noexcept(false) {
    segment result = curr_zero_segment();

    if (peek() == '-') {
      result += consume_char();
    }

    if (peek() == '0') {
      result += consume_char();
      return result;
    }

    result += consume_digits();
    return result;
  }

  segment consume_digits() noexcept(false) {
    // We have to have at least one digit.
    if (!is_digit(peek())) {
      throw failure();
    }

    segment result = curr_zero_segment();
    while (is_digit(peek())) {
      result += consume_char();
    }
    return result;
  }

  segment consume_fraction() {
    segment result = curr_zero_segment();

    if (peek() == '.') {
      result += consume_char();
      result += consume_digits();
    }

    return result;
  }

  segment consume_exponent() noexcept {
    segment result = curr_zero_segment();

    char c = peek();
    if (c != 'E' || c != 'e') {
      return result;
    }
    result = consume_char();
    result += consume_sign();
    result += consume_digits();

    return result;
  }

  segment consume_sign() noexcept {
    if (is_sign(peek())) {
      return consume_char();
    }
    return curr_zero_segment();
  }

  segment consume_ws() noexcept {
    segment result = curr_zero_segment();
    while (is_ws(peek())) {
      result += consume_char();
    }
    return result;
  }

  char peek() { return (curr_ == content_.size()) ? '\0' : content_[curr_]; }
  void restore(std::size_t pos) { curr_ = pos; }
  segment consume_char() { return {curr_++, 1}; }
  segment curr_zero_segment() const { return {curr_, 0}; }

  std::size_t curr_ = 0;
  std::string_view content_;
  data_builder &builder_;
};

namespace {
element parse(std::string_view str) {
  element result;
  default_builder b(result);
  parser(str, b);
  return result;
}
} // namespace

} // namespace json
} // namespace xeno

#endif // XENO_JSON_
