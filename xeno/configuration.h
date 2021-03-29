#include <any>
#include <filesystem>
#include <string_view>
#include <vector>

#include <xeno/exception.h>
#include <xeno/string.h>
#include <xeno/sys/file_descriptor.h>
#include <xeno/yaml.h>

// Only supposed to be used in main binary files. Don't set config for each
// library.

namespace xeno {
namespace configuration {

class flagstore {
public:
  template <typename T>
  void define_flag(std::string_view name, char short_name,
                   const T &default_value) {
    std::size_t index = flags_.size();
    if (!name.empty()) {
      names_.emplace(name, index);
    }
    if (short_name != 0) {
      short_names_.emplace(short_name, index);
    }
    flags_.emplace_back(default_value);
  }

  template <typename T1> void set_flag(std::string_view name, T1 &&val) {
    std::size_t index = find_index(name);
    flags_[index] = std::forward<T1>(val);
  }
  template <typename T1> void set_flag(char name, T1 &&val) {
    std::size_t index = find_index(name);
    flags_[index] = std::forward<T1>(val);
  }

  template <typename T> const T &get_flag(std::string_view name) {
    std::size_t index = find_index(name);
    if (flags_[index].type() != typeid(T)) {
      throw xeno::error(xeno::string::strcat("flag ", name, " has type ",
                                             flags_[index].type().name()));
    }

    T *result = std::any_cast<T>(&flags_[index]);
    return *result;
  }
  template <typename T> const T &get_flag(char name) {
    std::size_t index = find_index(name);
    if (flags_[index].type() != typeid(T)) {
      throw xeno::error(xeno::string::strcat("short flag ", name, " has type ",
                                             flags_[index].type().name()));
    }

    T *result = std::any_cast<T>(&flags_[index]);
    return *result;
  }

  std::vector<std::string_view>
  parse_from_args(std::span<std::string_view> argv) {
    std::vector<std::string_view> non_option_args;

    bool options_ended = false;
    for (std::size_t i = 0; i < argv.size(); ++i) {
      std::string_view arg = argv[i];

      if (options_ended) {
        non_option_args.push_back(arg);
        continue;
      }

      if (arg == "-") {
        // TODO: make a special flag for stdin and stdout in necessary.
        continue;
      }

      if (arg == "--") {
        options_ended = true;
        continue;
      }

      if (arg.starts_with("--")) {
        std::string_view option = arg.substr(2);
        std::vector<std::string_view> kv = xeno::string::split(option, '=');
        if (kv.size() == 1) {
          kv.push_back("");
        }

        if (kv.size() != 2) {
          throw xeno::error(xeno::string::strcat(
              "expecting key value pair for long option, but got ", arg));
        }
        parse_flag(kv[0], kv[1]);
        continue;
      }

      if (arg.starts_with('-')) {
        std::string_view option = arg.substr(1);

        if (option.size() != 1 || i == argv.size() - 1 ||
            argv[i + 1].starts_with('-')) {
          // Multiple options lumped in one.
          for (char c : option) {
            parse_flag(c);
          }
        } else {
          arg = argv[++i];
          parse_flag(option[0], arg);
        }
        continue;
      }

      non_option_args.push_back(arg);
    }
    return non_option_args;
  }

  void parse_from_yaml(const xeno::yaml::element &e) {
    const auto &m = e.get_mapping();
    for (const auto &[k, v] : m) {
      if (!v.has_string()) {
        throw xeno::error(xeno::string::strcat("no proper value for key ", k));
      }
      parse_flag(k, v.get_string());
    }
  }

private:
  template <typename T> void convert_flag_val(const T &v) {}

  template <typename T> void parse_flag(T name, std::string_view val = "") {
    std::size_t index = find_index(name);
    std::any &flag = flags_[index];

    if (flag.type() == typeid(std::string)) {
      flag = std::string(val);
    } else if (flag.type() == typeid(int)) {
      flag = std::stoi(std::string(val));
    } else if (flag.type() == typeid(bool)) {
      if (val.empty() || val == "true") {
        flag = true;
      } else if (val == "false") {
        flag = false;
      } else {
        throw xeno::error(
            xeno::string::strcat("flag ", name, " expects boolean value"));
      }
    } else {
      throw xeno::error(xeno::string::strcat(
          "flag ", name, " has unsupported type ", flag.type().name()));
    }
  }

  std::size_t find_index(std::string_view name) {
    auto pos = names_.find(std::string(name));
    if (pos == names_.end()) {
      throw xeno::error(xeno::string::strcat("undefined flag name ", name));
    }
    return pos->second;
  }

  std::size_t find_index(char name) {
    auto pos = short_names_.find(name);
    if (pos == short_names_.end()) {
      throw xeno::error(
          xeno::string::strcat("undefined flag short name ", name));
    }
    return pos->second;
  }

  std::map<std::string, std::size_t> names_;
  std::map<char, std::size_t> short_names_;
  std::vector<std::any> flags_;
}; // namespace configuration

std::vector<std::string_view> make_argv(std::span<char *> argv) {
  std::vector<std::string_view> result;
  result.reserve(argv.size());

  for (int i = 0; i < argv.size(); ++i) {
    result.emplace_back(argv[i]);
  }
  return result;
}

xeno::yaml::element config;
flagstore flags;

namespace {
std::filesystem::path find_conf_file(const std::filesystem::path &bin) {
  std::filesystem::path fn = bin.filename();
  std::filesystem::path dir = bin.parent_path();

  std::filesystem::path conf_path;

  std::filesystem::path conf_name = fn;
  conf_name.replace_extension("conf.yml");

  conf_path = dir / conf_name;

  if (std::filesystem::exists(conf_path))
    return conf_path;

  if (dir.filename() == ".out") {
    conf_path = dir.parent_path() / conf_name;
    conf_path = conf_path.lexically_normal();

    if (std::filesystem::exists(conf_path))
      return conf_path;
  }

  // TODO: check a central place.

  return std::filesystem::path();
}
} // namespace

std::vector<std::string_view> init_config(std::span<char *> system_argv) {
  std::vector<std::string_view> _ = make_argv(system_argv);
  std::span<std::string_view> argv(_);

  std::filesystem::path p = find_conf_file(argv[0]);
  if (!p.empty()) {
    std::string config_str = xeno::sys::file::open_as_string(p);
    config = xeno::yaml::parse(config_str);

    const auto &mapping = config.get_mapping();
    auto pos = mapping.find("flags");
    if (pos != mapping.end()) {
      flags.parse_from_yaml(pos->second);
    }
  }

  return flags.parse_from_args(argv.subspan(1));
}

} // namespace configuration
} // namespace xeno
