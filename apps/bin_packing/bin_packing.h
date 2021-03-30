#ifndef BIN_PACKING
#define BIN_PACKING

#include <random>
#include <sstream>

#include <xylo/nn.h>
#include <xylo/policy_gradient.h>

namespace bp {

constexpr std::size_t num_bins = 8;

using action = xylo::discrete_action<num_bins>;

struct observation {
  static std::size_t length() { return 4 * num_bins; }

  static constexpr std::pair<int, int> capacity{8, 8};

  observation(const std::pair<int, int> &bin_shape)
      : bins(num_bins, bin_shape), item{0, 0} {}

  std::string to_string() const {
    std::ostringstream oss;
    oss << "item: " << xeno::string::streamable(item) << "; ";
    oss << "bins: " << xeno::string::streamable(bins);
    return oss.str();
  }

  void to_vector(xylo::vector_view o) const {
    xylo::matrix_view m = xylo::fold<2>(o, {bins.size(), 4});

    for (std::size_t i = 0; i < bins.size(); ++i) {
      m[i][0] = float(bins[i].first) / capacity.first;
      m[i][1] = float(bins[i].second) / capacity.second;
      m[i][2] = float(item.first) / capacity.first;
      m[i][3] = float(item.second) / capacity.second;
    }
  }

  std::vector<std::pair<int, int>> bins;
  std::pair<int, int> item;
};

class environment : public xylo::environment<action, observation> {
public:
  static constexpr std::pair<int, int> capacity{8, 8};

  environment() : state_(std::make_unique<observation>(capacity)), dist_(0.4) {
    get_item();
  }
  void apply(const action &action, std::size_t id) override {
    std::pair<int, int> &bin = state_->bins[action.choice];
    std::pair<int, int> &item = state_->item;
    bin.first -= item.first;
    bin.second -= item.second;

    if (bin.first < 0 || bin.second < 0) {
      return;
    }

    get_item();
  }
  observation view(std::size_t id) const override { return *state_; }

  void reset(std::size_t id) override {
    state_ = std::make_unique<observation>(capacity);
    get_item();
  }

private:
  static constexpr std::pair<int, int> shape1{4, 2};
  static constexpr std::pair<int, int> shape2{1, 2};

  void get_item() {
    auto item = biased_coin_toss() ? shape1 : shape2;
    state_->item = item;
  }

  bool biased_coin_toss() { return dist_(xylo::default_generator()); }

  std::unique_ptr<observation> state_;
  std::bernoulli_distribution dist_;
};

class agent : public xylo::agent<action, observation> {
public:
  agent(const xylo::policy<action, observation> &p, environment &env,
        xylo::replay_buffer<action, observation> &rb)
      : xylo::agent<action, observation>(p, env, rb) {}

private:
  bool game_over(const observation &ob) override {
    for (const auto &bin : ob.bins) {
      if (bin.first < 0 || bin.second < 0) {
        return true;
      }
    }
    return false;
  }
  float get_reward(const observation &dummy, const observation &ob) override {
    if (game_over(ob))
      return 0;
    return 1;
  }
};

class pg_learner : public xylo::policy_gradient_learner<action, observation> {
public:
  pg_learner(xylo::replay_buffer<action, observation> &rb,
             xylo::model &action_model, xylo::optimizer &action_optimizer,
             float gamma = 0.99)
      : xylo::policy_gradient_learner<action, observation>(
            rb, action_model, action_optimizer, gamma) {}
};

class ac_learner : public xylo::actor_critic_learner<action, observation> {
public:
  ac_learner(xylo::replay_buffer<action, observation> &rb,
             xylo::model &action_model, xylo::optimizer &action_optimizer,
             xylo::model &value_model, xylo::optimizer &value_optimizer,
             float gamma = 0.99)
      : xylo::actor_critic_learner<action, observation>(
            rb, action_model, action_optimizer, value_model, value_optimizer,
            gamma) {}
};

class ppo_learner : public xylo::ppo_learner<action, observation> {
public:
  ppo_learner(xylo::replay_buffer<action, observation> &rb,
              xylo::model &action_model, xylo::optimizer &action_optimizer,
              xylo::model &value_model, xylo::optimizer &value_optimizer,
              float gamma = 0.99)
      : xylo::ppo_learner<action, observation>(rb, action_model,
                                               action_optimizer, value_model,
                                               value_optimizer, gamma) {}
};

class kl_ppo_learner : public xylo::kl_ppo_learner<action, observation> {
public:
  kl_ppo_learner(xylo::replay_buffer<action, observation> &rb,
                 xylo::model &action_model, xylo::optimizer &action_optimizer,
                 xylo::model &value_model, xylo::optimizer &value_optimizer,
                 float gamma = 0.99)
      : xylo::kl_ppo_learner<action, observation>(rb, action_model,
                                                  action_optimizer, value_model,
                                                  value_optimizer, gamma) {}
};

} // namespace bp

#endif
