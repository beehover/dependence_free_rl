#include "xylo/nn.h"
#include <xeno/sys/thread.h>

#include <xylo/bin_packing.h>
#include <xylo/rl.h>

using trajectory = xylo::trajectory<bp::action, bp::observation>;

int main() {
  xylo::model action_model;
  action_model.add_layer(
      std::make_unique<xylo::full_layer>(bp::num_bins * 4, 256));
  action_model.add_layer(std::make_unique<xylo::relu_activation>());
  action_model.add_layer(std::make_unique<xylo::full_layer>(256, 128));
  action_model.add_layer(std::make_unique<xylo::relu_activation>());
  action_model.add_layer(std::make_unique<xylo::full_layer>(128, bp::num_bins));
  action_model.add_layer(std::make_unique<xylo::softmax_cross_entropy_layer>());
  xylo::sgd_optimizer action_optimizer(action_model, 1e-4);

  xylo::replay_buffer<bp::action, bp::observation> replay_buffer;

  const int num_workers = 4;
  const int episodes_per_worker = 4;

  std::vector<bp::environment> envs;
  std::vector<bp::agent> agents;

  xylo::policy_gradient_policy<bp::action, bp::observation> policy(
      action_model);

  envs.reserve(num_workers);
  agents.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    envs.emplace_back();
    agents.emplace_back(policy, envs[i], replay_buffer);
  }

  bp::pg_learner learner(replay_buffer, action_model, action_optimizer, 0.99);

  std::list<xeno::sys::thread> threads;
  for (int i = 0; i < num_workers; ++i) {
    threads.emplace_back(xeno::string::strcat("worker", i));
  }

  float max_reward = 0;
  for (int steps = 0;; ++steps) {
    int i = 0;
    for (auto &t : threads) {
      bp::agent &agent = agents[i++];
      t.run([&]() {
        for (int i = 0; i < episodes_per_worker; ++i) {
          agent.play_one_episode();
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }

    if (steps % 100 == 0) {
      float total_rewards = 0;
      for (xylo::td<bp::action, bp::observation> traj :
           replay_buffer.sample_td()) {
        total_rewards += traj.size();
      }
      float avg_rewards =
          total_rewards * 1.0f / num_workers / episodes_per_worker;
      lg() << "avg rewards " << steps << " :" << avg_rewards;
    }

    learner.step();

    replay_buffer.forget();
  }

  return 0;
}
