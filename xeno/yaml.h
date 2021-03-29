#ifndef XENO_YAML_
#define XENO_YAML_

#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <xeno/string.h>

namespace xeno::yaml {

// Simplified yaml.
// 0. One document in a file.
// 1. No tags or anchors are supported.
// 2. Not compatible with json.
// 3. All value types are treated as strings.
// 4. All characters are ASCII and no escaping is supported.
// 5. For map types, if a value is scalar, it is placed on
// the same line as the key. Otherwise a new line is required.
// 6. New line must be '\n'.
// 7. Sequences must have strings as elements.
// 8. Not empty node allowed.
// 9. Tabs are not considered space or indentation. It's best they are not used,
// but we don't check for its existence either.
class element {
public:
  using seq_t = std::vector<std::string>;
  using mapping_t = std::map<std::string, element>;
  using null_t = std::monostate;
  using val_t = std::variant<null_t, std::string, seq_t, mapping_t>;

  bool has_string() const {
    return std::holds_alternative<std::string>(value_);
  }
  bool has_mapping() const { return std::holds_alternative<mapping_t>(value_); }
  bool has_sequence() const { return std::holds_alternative<seq_t>(value_); }

  bool is_null() const { return std::holds_alternative<null_t>(value_); }

  std::string_view get_string() const { return std::get<std::string>(value_); }
  const seq_t &get_sequence() const { return std::get<seq_t>(value_); }
  seq_t &get_mutable_sequence() {
    if (!has_sequence())
      value_.emplace<seq_t>();
    return std::get<seq_t>(value_);
  }
  const mapping_t &get_mapping() const { return std::get<mapping_t>(value_); }
  mapping_t &get_mutable_mapping() {
    if (!has_mapping())
      value_.emplace<mapping_t>();
    return std::get<mapping_t>(value_);
  }

  void clear() { value_ = std::monostate(); }
  void set_string(std::string_view str) { value_ = std::string(str); }
  void set_string(std::string &&str) { value_ = std::move(str); }

  element &operator[](std::string_view key) {
    return get_mutable_mapping()[std::string(key)];
  }
  const element &operator[](std::string_view key) const {
    const mapping_t &mapping = get_mapping();
    mapping_t::const_iterator iter = mapping.find(std::string(key));
    if (iter == mapping.end()) {
      throw std::out_of_range(
          xeno::string::strcat("No key ", key, " in mapping"));
    }
    return iter->second;
  }

  std::string to_string() const {
    std::ostringstream oss;
    to_string_helper(oss, *this, 0);
    return oss.str();
  }

private:
  static void indent_oss(std::ostringstream &oss, int64_t indent) {
    for (int i = 0; i < indent; ++i) {
      oss << ' ';
    }
  }

  static void to_string_helper(std::ostringstream &oss, const element &val,
                               int64_t indent) {
    if (val.has_string()) {
      indent_oss(oss, indent);
      oss << val.get_string() << '\n';
    } else if (val.has_sequence()) {
      for (const auto entry : val.get_sequence()) {
        indent_oss(oss, indent);
        oss << "- " << entry << '\n';
      }
    } else if (val.has_mapping()) {
      for (const auto &[key, sub_val] : val.get_mapping()) {
        indent_oss(oss, indent);
        oss << key << ':';
        if (sub_val.has_string()) {
          oss << ' ' << sub_val.get_string() << '\n';
        } else {
          oss << '\n';
          to_string_helper(oss, sub_val, indent + 2);
        }
      }
    }
  }

  val_t value_;
};

namespace {

struct line_state {
  std::string to_string() {
    return string::strcat("\nindent: ", indent, "\nkey: ", key,
                          "\nval: ", value, "\nseq: ", seq_entry);
  }

  // A line may have:
  // 1. A key and a value.
  // 2. A key only.
  // 3. A value only.
  // 4. A sequence entry only.
  int64_t indent = -1;
  std::string_view key = "";
  std::string_view value = "";
  std::string_view seq_entry = "";
};

std::string_view strip(std::string_view str) {
  std::size_t pos = str.find_first_not_of(' ');
  if (pos == std::string_view::npos) {
    // Empty string. Directly return.
    return str.substr(0, 0);
  }

  str.remove_prefix(pos);
  pos = str.find_last_not_of(' ');
  return str.substr(0, pos + 1);
}

std::vector<line_state> convert_states(std::string_view content) {
  std::vector<std::string_view> lines = string::split(content, '\n');
  std::vector<line_state> results;
  for (std::string_view line : lines) {
    // Strip comments first.
    std::size_t pond_pos = line.find_first_of('#');
    line = line.substr(0, pond_pos);

    // Get the indent level.
    std::size_t indent = line.find_first_not_of(' ');

    // If we can't get the indent level, it must be an empty line. Bail now.
    if (indent == std::string_view::npos)
      continue;

    // Now strip the indentation.
    line.remove_prefix(indent);

    results.emplace_back();
    line_state &curr_line_state = results.back();
    curr_line_state.indent = indent;

    if (line.starts_with('-')) {
      // Seq entry.
      line.remove_prefix(1);
      curr_line_state.seq_entry = strip(line);
    } else {
      std::size_t pos = line.find(':');
      if (pos != std::string_view::npos) {
        // At least a key.
        curr_line_state.key = strip(line.substr(0, pos));
        curr_line_state.value = strip(line.substr(pos + 1));
      } else {
        // Just a value.
        curr_line_state.value = strip(line);
      }
    }
  }
  return results;
}

element build_element(std::vector<line_state> line_states) {
  element result;
  std::vector<std::pair<line_state, element &>> hierarchy_stack;

  hierarchy_stack.emplace_back(line_state{-1, "", "", ""}, result);
  for (const auto &curr_line : line_states) {
  maybe_pop:
    // We might have a series of pending mappings to resolve.
    // We pop them one by one until we see an entry that has less
    // indentation than us (and thus our true parent).
    auto &[last_line, last_element] = hierarchy_stack.back();
    if (curr_line.indent <= last_line.indent) {
      hierarchy_stack.pop_back();
      goto maybe_pop;
    }

    // We're here so we "last_line" holds our parent.
    if (!curr_line.seq_entry.empty()) {
      last_element.get_mutable_sequence().emplace_back(curr_line.seq_entry);
      continue;
    }

    if (!curr_line.key.empty()) {
      if (!curr_line.value.empty()) {
        last_element[curr_line.key].set_string(curr_line.value);
      } else {
        // No value on this line. Let's push the current element on the stack.
        hierarchy_stack.emplace_back(curr_line, last_element[curr_line.key]);
      }
    } else {
      // Gotta be a lone value.
      last_element.set_string(curr_line.value);
    }
  }
  return result;
}
} // namespace

xeno::yaml::element parse(std::string content) {
  std::vector<xeno::yaml::line_state> line_states =
      xeno::yaml::convert_states(content);
  return xeno::yaml::build_element(line_states);
}
} // namespace xeno::yaml

#endif // XENO_YAML_
