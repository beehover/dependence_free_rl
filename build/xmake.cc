#include <stdlib.h>

#include <map>

#include <xeno/logging.h>
#include <xeno/string.h>
#include <xeno/sys/file_descriptor.h>
#include <xeno/sys/filesystem.h>
#include <xeno/time.h>
#include <xeno/yaml.h>

namespace fs = std::filesystem;
namespace xmake {

constexpr char xmake_file[] = "xmake.yml";
constexpr char repo_mark[] = ".git";
constexpr char outdir[] = ".out";

fs::path get_repo_abs_path() {
  fs::path d = fs::current_path();
  while (!exists(d / repo_mark)) {
    d = d.parent_path();
  }
  return d;
}
fs::path repo_abs_path = get_repo_abs_path();

fs::path curr_rel_path() {
  fs::path d = fs::current_path();
  // return fs::relative(d, repo_abs_path);
  return d.lexically_relative(repo_abs_path);
}
fs::path prefix_to_rel_path(const fs::path &prefix) {
  return prefix.lexically_relative("//");
}
fs::path rel_path_to_prefix(const fs::path &rel_path) {
  return fs::path("//") / rel_path;
}
fs::path curr_prefix() { return rel_path_to_prefix(curr_rel_path()); }

fs::path rel_to_abs_path(const fs::path &rel_path) {
  return repo_abs_path / rel_path;
}
fs::path abs_to_rel_path(const fs::path &abs_path) {
  return abs_path.lexically_relative(repo_abs_path);
}

fs::path prefix_to_abs_path(const fs::path &prefix) {
  return rel_to_abs_path(prefix_to_rel_path(prefix));
}

fs::path out_abs_path(const fs::path &abs_path) { return abs_path / outdir; }

namespace {
template <typename K, typename V> struct kv_resolution {
public:
  using func_t = std::function<V(const K &)>;
  kv_resolution() = default;
  kv_resolution(const func_t &f) : func_(f) {}
  kv_resolution(const func_t &f, const K &k, bool resolve_now = false)
      : func_(f), key_(k) {
    if (resolve_now)
      resolve();
  }
  void resolve() { *val_ = func_(key_); }

  void set_key(const K &key) { key_ = key; }
  void set_key(K &&key) { key_ = std::move(key); }

  K get_key() const { return key_; }
  V get_val() const { return *val_; }

  void set_func(const func_t &f) { func_ = f; }
  void set_func(func_t &&f) { func_ = std::move(f); }

private:
  K key_;
  std::optional<V> val_;
  func_t func_;
};
using fts = kv_resolution<fs::path, xeno::time::point>;

struct build_target;
// using fdep =
//    std::pair<fs::path, std::optional<std::reference_wrapper<build_target>>>;
using fdep = kv_resolution<fs::path, std::reference_wrapper<build_target>>;

xeno::time::point resolve_time(const fs::path &p) {
  return fs::exists(p) ? xeno::sys::modification_time(p) : xeno::time::epoch();
};

struct build_target {
  fts file;
  std::string rule = "c++";
  bool main = false;
  std::vector<fts> srcs;
  std::vector<fts> hdrs;
  std::vector<fdep> deps;
  std::vector<fdep> data;
  std::vector<std::string> lopts;
  std::vector<std::string> gopts;
  bool built = false;
};

xeno::yaml::element get_value(std::string_view key,
                              const xeno::yaml::element::mapping_t &mapping) {
  auto iter = mapping.find(std::string(key));
  if (iter == mapping.end()) {
    return xeno::yaml::element();
  }

  return iter->second;
}
} // namespace

std::map<fs::path, build_target> target_map;

std::reference_wrapper<build_target>
resolve_dependency(const fs::path &target_prefix);

void load_prefix(const fs::path &prefix) {
  std::string str = xeno::sys::file::open_as_string(prefix_to_abs_path(prefix) /
                                                    xmake::xmake_file);
  auto raw_targets = xeno::yaml::parse(str).get_mapping();

  for (const auto &[name, entry] : raw_targets) {
    build_target &target = target_map[prefix / name];
    target.file.set_key(out_abs_path(prefix_to_abs_path(prefix)) / name);
    target.file.set_func(resolve_time);

    auto attributes = entry.get_mapping();

    auto main = get_value("main", attributes);
    target.main = main.has_string() && main.get_string() == "true";

    if (!target.main && target.rule == "c++")
      target.file.set_key(target.file.get_key().replace_extension(".a"));
    target.file.resolve();

    auto srcs = get_value("srcs", attributes);
    if (srcs.has_sequence()) {
      for (const auto &src : srcs.get_sequence()) {
        auto src_file = prefix_to_abs_path(prefix) / src;
        target.srcs.emplace_back(resolve_time, src_file, true);
      }
    }
    auto hdrs = get_value("hdrs", attributes);
    if (hdrs.has_sequence()) {
      for (const auto &hdr : hdrs.get_sequence()) {
        auto hdr_file = prefix_to_abs_path(prefix) / hdr;
        target.hdrs.emplace_back(resolve_time, hdr_file, true);
      }
    }
    auto deps = get_value("deps", attributes);
    if (deps.has_sequence()) {
      for (const auto &dep : deps.get_sequence()) {
        auto back = target.deps.emplace_back(resolve_dependency, dep, false);
      }
    }
  }
}

std::reference_wrapper<build_target>
resolve_dependency(const fs::path &target_prefix) {
  if (!target_map.contains(target_prefix)) {
    // Target_prefix is the prefix of the target. Parent is the prefix of the
    // package.
    std::cerr << "[Loading] " << target_prefix.parent_path().string()
              << std::endl;
    load_prefix(target_prefix.parent_path());

    // If we still can't resolve the target prefix, user must have specified
    // the wrong target.
    if (!target_map.contains(target_prefix)) {
      throw xeno::error(
          xeno::string::strcat("can't load for target ", target_prefix));
    }
  }

  return target_map[target_prefix];
}

void resolve_all_dependencies(fdep &dep) {
  dep.resolve();
  for (auto &subdep : dep.get_val().get().deps) {
    resolve_all_dependencies(subdep);
  }
}

namespace {
std::string CC = "clang++";
std::string AR = "ar";
std::string LINKER = "ld";
std::string OPT = "-O3";
std::string G = "-g0";
std::string STD = "-std=c++20";
std::string AVX = "-mavx";
std::string FAST_MATH = "-ffast-math";
std::string INCLUDE = xeno::string::strcat("-I", repo_abs_path.string());
// std::string INCLUDE2 = "-I/usr/local/cuda-11.2/include/";
std::string PTHREAD = "-pthread";
// std::string LIBRARY = "-L/usr/local/cuda-11.2/lib64/";
std::string OPENSSL = "-lssl";
std::string CRYPTO = "-lcrypto";
std::string ATOMIC = "-latomic";
// std::string CUDA_RUNTIME = "-lcudart";
} // namespace

int build_dot_c(const fs::path &target, const fs::path &src) {
  std::vector<std::string> cmd_vec{CC, OPT, STD, AVX, FAST_MATH};
  cmd_vec.emplace_back(INCLUDE);
  // cmd_vec.emplace_back(INCLUDE2);
  cmd_vec.emplace_back("-c");
  cmd_vec.emplace_back(src.string());
  cmd_vec.emplace_back("-o");
  cmd_vec.emplace_back(target.string());
  std::string cmd = xeno::string::join(cmd_vec, ' ');
  std::cerr << "[CC] " << cmd << std::endl;
  return system(cmd.c_str());
}

int link_binary(const fs::path &target, const std::vector<fs::path> &srcs) {
  std::vector<std::string> cmd_vec{CC,
                                   PTHREAD,
                                   OPENSSL,
                                   CRYPTO,
                                   ATOMIC, /*CUDA_RUNTIME,*/
                                   /*LIBRARY,*/ "-o",
                                   target};
  for (const auto &src : srcs) {
    cmd_vec.emplace_back(src.string());
  }
  std::string cmd = xeno::string::join(cmd_vec, ' ');
  std::cerr << "[LK] " << cmd << std::endl;
  return system(cmd.c_str());
}

int link_archive(const fs::path &target, const std::vector<fs::path> &srcs) {
  std::vector<std::string> cmd_vec{AR, "rcsuUPT", target};
  for (const auto &src : srcs) {
    cmd_vec.emplace_back(src.string());
  }
  std::string cmd = xeno::string::join(cmd_vec, ' ');
  std::filesystem::remove(target);
  std::cerr << "[AR] " << cmd << std::endl;
  return system(cmd.c_str());
}

std::vector<fs::path> merge_files(const std::vector<fs::path> &v1,
                                  const std::vector<fs::path> &v2) {
  std::vector<fs::path> result;
  for (const auto &item : v1) {
    result.push_back(item);
  }
  for (const auto &item : v2) {
    result.push_back(item);
  }
  return result;
}

int build_single_cpp_target(const build_target &target) {
  std::vector<fs::path> dot_o_files;
  for (const auto &src : target.srcs) {
    fs::path src_name = src.get_key();
    fs::path dot_o = target.file.get_key().parent_path() /
                     src_name.filename().replace_extension(".o");
    dot_o_files.push_back(dot_o);
    if (int ret = build_dot_c(dot_o, src_name)) {
      return ret;
    }
  }

  std::vector<fs::path> dot_a_files;
  for (const auto &dep : target.deps) {
    dot_a_files.emplace_back(dep.get_val().get().file.get_key());
  }

  std::vector<fs::path> objs = merge_files(dot_o_files, dot_a_files);

  if (target.main) {
    if (int ret = link_binary(target.file.get_key(), objs)) {
      return ret;
    }
  } else {
    if (int ret = link_archive(target.file.get_key(), objs)) {
      return ret;
    }
  }
  return 0;
}

int build_single_target(const build_target &target) {
  if (target.rule == "c++") {
    return build_single_cpp_target(target);
  }
  return -1;
}

int build(build_target &target, bool force = false) {
  for (auto &dep : target.deps) {
    if (int ret = build(dep.get_val().get(), force); ret != 0) {
      return ret;
    }
  }

  bool all_srcs_older = std::ranges::all_of(target.srcs, [&](auto src) {
    return target.file.get_val() > src.get_val();
  });
  bool all_hdrs_older = std::ranges::all_of(target.hdrs, [&](auto hdr) {
    return target.file.get_val() > hdr.get_val();
  });
  bool all_deps_older = std::ranges::all_of(target.deps, [&](auto dep) {
    return target.file.get_val() > dep.get_val().get().file.get_val();
  });

  if (!force && all_srcs_older && all_hdrs_older && all_deps_older) {
    return 0;
  }

  fs::create_directories(target.file.get_key().parent_path());
  int ret = build_single_target(target);
  target.file.resolve();
  return ret;
}

} // namespace xmake

std::vector<std::string_view> get_args(int argc, const char *argv[]) {
  std::vector<std::string_view> result;
  for (int i = 0; i < argc; ++i) {
    result.emplace_back(argv[i]);
  }
  return result;
}

int main(int argc, const char *argv[]) {
  std::vector<std::string_view> args = get_args(argc, argv);

  if (args.size() == 1) {
    lg(lg::error) << "need a target";
    return -EINVAL;
  }

  std::string_view name = args[1];
  fs::path top_target_prefix = xmake::curr_prefix() / name;
  xmake::fdep top_fdep(xmake::resolve_dependency, top_target_prefix, false);

  xmake::resolve_all_dependencies(top_fdep);
  int ret = xmake::build(top_fdep.get_val().get());
  exit(ret == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
