#include <xeno/sys/thread.h>

#include <xylo/nn.h>
#include <xylo/rl.h>

#include <apps/bin_packing/bin_packing.h>

using trajectory = xylo::trajectory<bp::action, bp::observation>;

class bestfit_policy : public xylo::policy<bp::action, bp::observation> {
  bp::action react(const bp::observation &state) const override {
    bp::action a;

    xylo::vector scores({bp::num_bins});

    for (std::size_t i = 0; i < bp::num_bins; ++i) {
      if (state.item.first > state.bins[i].first ||
          state.item.second > state.bins[i].second) {
        scores[i] = -1;
        continue;
      }

      scores[i] = float(state.item.first) / state.bins[i].first +
                  float(state.item.second) / state.bins[i].second;
    }

    a.from_vector_deterministic(scores);
    return a;
  }
};

int main() {
  constexpr std::size_t num_episodes = 10000;
  for (std::size_t steps = 0; steps <= 100; ++steps) {
    bestfit_policy policy;
    bp::environment env;
    xylo::replay_buffer<bp::action, bp::observation> rb;
    bp::agent agent(policy, env, rb);
    for (int i = 0; i < num_episodes; ++i) {
      agent.play_one_episode();
    }
    auto experience = rb.sample_td();
    lg() << "round " << steps << " "
         << xylo::total_rewards<bp::action, bp::observation>(experience) /
                num_episodes;

    rb.forget();
  }
  return 0;
}
