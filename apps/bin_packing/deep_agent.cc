#include "xylo/tensor.h"
#include <xeno/sys/file_descriptor.h>
#include <xeno/sys/thread.h>

#include <xylo/nn.h>
#include <xylo/rl.h>

#include <apps/bin_packing/bin_packing.h>

int main() {
  xylo::model action_model;
  action_model.add_layer(std::make_unique<xylo::convolution1d_1_layer>(4, 128));
  action_model.add_layer(std::make_unique<xylo::relu_activation>());
  action_model.add_layer(
      std::make_unique<xylo::convolution1d_1_layer>(128, 64));
  action_model.add_layer(std::make_unique<xylo::relu_activation>());
  action_model.add_layer(std::make_unique<xylo::convolution1d_1_layer>(64, 1));

  // wieghts.10 is good
  // wieghts.20 is good
  xeno::sys::mmap f = xeno::sys::mmap<float>("weights.20");
  xylo::vector_view v = xylo::borrow_vector(f.span());
  action_model.set_parameters(v);

  constexpr std::size_t num_episodes = 10000;
  for (std::size_t steps = 0; steps <= 1000; ++steps) {
    xylo::policy_gradient_deterministic_policy<bp::action, bp::observation>
        policy(action_model);
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
