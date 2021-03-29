#ifndef XENO_SYS_FILESYSTEM
#define XENO_SYS_FILESYSTEM

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <filesystem>

#include <xeno/time.h>

namespace {
namespace fs = std::filesystem;
}

namespace xeno::sys {
xeno::time::point modification_time(const fs::path &p) {
  struct ::stat stat_buf;
  std::timespec ts;
  ::stat(p.c_str(), &stat_buf);
  return xeno::time::point(stat_buf.st_mtim);
}
} // namespace xeno::sys

#endif // XENO_SYS_FILESYSTEM
