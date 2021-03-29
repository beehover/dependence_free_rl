#include <apps/bin_packing/bin_packing.h>

int main() {
  for (std::size_t steps = 0; steps <= 100; ++steps) {
    xylo::random_policy<bp::num_bins, bp::observation> policy;
    bp::environment env;
    xylo::replay_buffer<bp::action, bp::observation> rb;
    bp::agent agent(policy, env, rb);
    for (int i = 0; i < 100; ++i) {
      agent.play_one_episode();
    }
    auto experience = rb.sample_td();
    lg() << "round " << steps << " "
         << xylo::total_rewards<bp::action, bp::observation>(experience) /
                100.0;

    rb.forget();
  }

  return 0;
}
