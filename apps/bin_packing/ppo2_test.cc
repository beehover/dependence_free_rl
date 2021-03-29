#include <xeno/sys/file_descriptor.h>
#include <xeno/sys/io.h>
#include <xeno/sys/thread.h>

#include <xylo/rl.h>

#include <apps/bin_packing/bin_packing.h>

using trajectory = xylo::trajectory<bp::action, bp::observation>;

int main() {
  xylo::model action_model;
  action_model.add_layer(std::make_unique<xylo::convolution1d_1_layer>(4, 128));
  action_model.add_layer(std::make_unique<xylo::relu_activation>());
  action_model.add_layer(
      std::make_unique<xylo::convolution1d_1_layer>(128, 64));
  action_model.add_layer(std::make_unique<xylo::relu_activation>());
  action_model.add_layer(std::make_unique<xylo::convolution1d_1_layer>(64, 1));
  action_model.add_layer(std::make_unique<xylo::softmax_layer>());
  xylo::sgd_optimizer action_optimizer(action_model, 1e-4, 1e-5);

  xylo::model value_model;
  value_model.add_layer(
      std::make_unique<xylo::full_layer>(4 * bp::num_bins, 64));
  value_model.add_layer(std::make_unique<xylo::relu_activation>());
  value_model.add_layer(std::make_unique<xylo::full_layer>(64, 32));
  value_model.add_layer(std::make_unique<xylo::relu_activation>());
  value_model.add_layer(std::make_unique<xylo::full_layer>(32, 1));
  xylo::sgd_optimizer value_optimizer(value_model, 1e-5);

  xylo::replay_buffer<bp::action, bp::observation> replay_buffer;

  constexpr int num_workers = 16;
  constexpr int steps_per_worker = 8;

  std::vector<bp::environment> envs;
  std::vector<bp::agent> agents;

  envs.reserve(num_workers);
  agents.reserve(num_workers);
  xylo::policy_gradient_policy<bp::action, bp::observation> policy(
      action_model);
  for (int i = 0; i < num_workers; ++i) {
    envs.emplace_back();
    agents.emplace_back(policy, envs[i], replay_buffer);
  }

  bp::kl_ppo_learner learner(replay_buffer, action_model, action_optimizer,
                             value_model, value_optimizer, 0.99);

  std::list<xeno::sys::thread> threads;
  for (int i = 0; i < num_workers; ++i) {
    threads.emplace_back(xeno::string::strcat("worker", i));
  }
  float max_reward = 0;

  int weights_file_no = 0;

  for (int steps = 0;; ++steps) {
    int i = 0;
    for (auto &t : threads) {
      bp::agent &agent = agents[i++];
      t.run([&]() { agent.play_steps(steps_per_worker); });
    }
    for (auto &t : threads) {
      t.join();
    }

    learner.step();

    replay_buffer.forget();

    if (steps % 100 == 0) {
      xylo::policy_gradient_deterministic_policy<bp::action, bp::observation>
          policy(action_model);
      bp::environment env;
      xylo::replay_buffer<bp::action, bp::observation> rb;
      bp::agent agent(policy, env, rb);
      std::size_t num_episodes = steps > 10000 ? 10000 : 100;
      for (int i = 0; i < num_episodes; ++i) {
        agent.play_one_episode();
      }
      auto experience = rb.sample_td();
      float avg_rewards =
          xylo::total_rewards<bp::action, bp::observation>(experience) /
          num_episodes;
      lg() << "round " << steps << " " << avg_rewards;

      if (avg_rewards > 26.5) {
        std::string filename =
            xeno::string::strcat("weights.", weights_file_no++);
        xeno::sys::file f = xeno::sys::file::open_to_append(filename);
        xeno::sys::buffered_blocking_io<xeno::sys::file> io(f);
        xylo::vector v = action_model.parameters();
        std::span<float> s(v.begin(), v.end());
        io.assured_write(std::as_bytes(s));
      }

      rb.forget();
    }
  }

  return 0;
}
