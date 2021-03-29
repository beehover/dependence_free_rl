#include <sys/types.h>
#include <unistd.h>

#include <string>

#include <xeno/logging.h>
#include <xeno/sys/thread.h>

namespace xeno::logging {
extern thread_local std::string thread_name;
}

namespace xeno::sys {

namespace {
void set_local_name(std::string_view s) {
  // We set logging thread name here in order to avoid making logging depend on
  // us.
  xeno::logging::thread_name = s;
}
} // namespace

void thread::join() {
  int ret = pthread_join(pid_, nullptr);
  pid_ = 0;
}

void *thread::execute_closure(void *p) {
  thread *t = reinterpret_cast<thread *>(p);
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
  set_local_name(t->name_);
  std::invoke(t->closure_);
  return nullptr;
}

} // namespace xeno::sys
