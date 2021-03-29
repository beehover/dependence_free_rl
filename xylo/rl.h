#ifndef XYLO_RL_
#define XYLO_RL_

#include <atomic>
#include <functional>
#include <list>
#include <mutex>
#include <span>
#include <vector>

#include <xylo/nn.h>
#include <xylo/tensor.h>

namespace xylo {

template <typename T> vector to_vector(const T &t) {
  vector result({t.length()});
  t.to_vector(result);
  return result;
}

template <std::size_t range> struct discrete_action {
  static std::size_t cardinality() { return range; }
  std::size_t choice;
  std::optional<vector> distrib;

  void from_vector(vector_view a) {
    choice = discrete_distribution(a);
    distrib = a;
  }
  void from_vector_deterministic(vector_view a) { choice = argmax(a); }

  void gradient_log(vector_view input, vector_view output,
                    float advantage) const {
    if (input.size() != range || output.size() != range)
      throw std::exception();

    output = 0;
    float log_action_grad = 1 / input[choice];
    float weighted_grad = log_action_grad * advantage * -1;
    float importance_grad = input[choice] / (*distrib)[choice] * weighted_grad;
    output[choice] = importance_grad;
  }

  void softmax_gradient_log(vector_view input, vector_view output,
                            float advantage) const {
    if (input.size() != range || output.size() != range)
      throw std::exception();

    output = input * advantage;
    output[choice] -= advantage;
  }

  void clipped_gradient(vector_view input, vector_view output,
                        float advantage) const {
    constexpr float epsilon = 0.2;

    if (input.size() != range || output.size() != range)
      throw std::exception();

    output = 0;
    float ratio = input[choice] / (*distrib)[choice];

    float clipped_ratio = ratio;
    if (ratio > (1 + epsilon)) {
      clipped_ratio = 1 + epsilon;
    } else if (ratio < (1 - epsilon)) {
      clipped_ratio = 1 - epsilon;
    }

    float importance_grad =
        std::min(clipped_ratio * advantage, ratio * advantage) * -1;
    output[choice] = importance_grad / input[choice];
  }
};

struct continuous_action {
  static std::size_t cardinality() { return 1; }

  float action;
  float mean;
  float stddev = 1;

  void from_vector(vector_view a) {
    vector result({1});
    mean = a[0];
    normal_distribution(mean, stddev, result);
    action = result[0];
  }
  void gradient_log(vector_view input, vector_view output, float reward,
                    float o_value) const {
    if (input.size() != 1 || output.size() != 1)
      throw std::exception();

    float log_action_grad = (action - input[0]) / (stddev * stddev);
    float weighted_grad = log_action_grad * (reward / o_value - 1) * -1;
    float normalized_input_action_diff = (action - input[0]) / stddev;
    float normalized_action_diff = (action - mean) / stddev;
    float importance_grad =
        ::exp(-0.5 *
              (normalized_input_action_diff * normalized_input_action_diff -
               normalized_action_diff * normalized_action_diff)) *
        weighted_grad;
    output[0] = importance_grad;
  }

  void clipped_gradient(vector_view input, vector_view output, float reward,
                        float o_value) const {}
};

template <typename A, typename S> struct transition {
  transition() = default;
  transition(const S &prev, A &&a, float r, S &&curr)
      : action(std::move(a)), reward(r), end_state(std::move(curr)) {}

  const S *start_state = nullptr;
  A action;
  float reward;
  S end_state;
}; // namespace xylo

// We own all actions and states added.
template <typename A, typename S> struct trajectory {
  trajectory(S &&o) : opening(std::move(o)), frozen(false) {}

  void add_transition(A &&a, float r, S &&curr) {
    transitions.emplace_back(last_state(), std::move(a), r, std::move(curr));
  }

  const S &last_state() {
    if (transitions.empty()) {
      return opening;
    }

    return transitions.back().end_state;
  }

  std::size_t size() const { return transitions.size(); }

  void fill_reference() {
    if (transitions.empty())
      return;

    auto pos = transitions.begin();
    pos->start_state = &opening;

    for (auto last_pos = pos++; pos != transitions.end(); last_pos = pos++) {
      pos->start_state = &last_pos->end_state;
      last_pos = pos;
    }
  }

  void freeze() {
    frozen = true;
    fill_reference();
  }

  S opening;
  std::list<transition<A, S>> transitions;
  bool frozen;
};

template <typename A, typename S> class environment {
public:
  virtual ~environment() = default;

  virtual void apply(const A &action, std::size_t id) = 0;
  virtual S view(std::size_t id) const = 0;
  virtual void reset(std::size_t id) = 0;
};

// Temporal differences
template <typename A, typename S> class td {
public:
  using container = std::list<transition<A, S>>;
  td(const trajectory<A, S> &traj)
      : frozen_(traj.frozen), size_(traj.transitions.size()),
        begin_(traj.transitions.begin()), end_(traj.transitions.end()),
        back_(traj.transitions.back()) {}

  typename container::const_iterator begin() const { return begin_; }
  typename container::const_iterator end() const { return end_; }

  std::size_t size() const { return size_; }

  bool frozen() const { return frozen_; }

  const transition<A, S> &front() const { return *begin_; }
  const transition<A, S> &back() const { return back_; }

private:
  bool frozen_;
  std::size_t size_;
  const typename container::const_iterator begin_;
  const typename container::const_iterator end_;
  const transition<A, S> &back_;
};

template <typename A, typename S>
float total_rewards(const std::vector<td<A, S>> &experience) {
  float result = 0;
  for (const auto &traj : experience) {
    for (const auto &transition : traj) {
      result += transition.reward;
    }
  }
  return result;
}

template <typename A, typename S>
using transition_ref = std::reference_wrapper<transition<A, S>>;

template <typename A, typename S> class replay_buffer {
public:
  trajectory<A, S> &emplace_trajectory(S &&s) {
    std::lock_guard l(mutex_);
    trajectories_.emplace_back(std::move(s));
    return trajectories_.back();
  }

  // TODO: parameters are not implemented yet.
  std::vector<td<A, S>> sample_td(std::size_t n = -1,
                                  std::size_t max_length = -1) {
    std::vector<td<A, S>> result;

    for (trajectory<A, S> &traj : trajectories_) {
      traj.fill_reference();
      result.emplace_back(traj);
      if (traj.size() == 0) {
        continue;
      }
    }
    return result;
  }

  std::vector<transition_ref<A, S>> sample_transitions(std::size_t n) {
    std::vector<transition_ref<A, S>> result;
    result.reserve(n);

    // TODO: implement the sampling
    std::size_t total = 0;
    for (trajectory<A, S> &traj : trajectories_) {
      total += traj.size();
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, total - 1);

    for (std::size_t i = 0; i < n; ++i) {
      std::size_t index = distrib(gen);

      std::size_t start = 0;
      transition<A, S> *p_trans = nullptr;
      for (trajectory<A, S> &traj : trajectories_) {
        std::size_t end = start + traj.size();
        if (index >= start && index < end) {
          std::size_t curr = start;
          for (transition<A, S> &trans : traj.transitions) {
            if (curr++ == index) {
              p_trans = &trans;
              break;
            }
          }
          break;
        }
        start = end;
      }
      result.emplace_back(*p_trans);
    }
    return result;
  }

  void forget() {
    std::vector<trajectory<A, S>> left;
    for (auto pos = trajectories_.begin(); pos != trajectories_.end();) {
      auto &trajectory = *pos;

      if (trajectory.frozen) {
        // Forget the whole trajectory.
        pos = trajectories_.erase(pos);
        continue;
      }

      // Forget everything except the last state, so that further transitions
      // can come in later.
      trajectory.opening = std::move(trajectory.transitions.back().end_state);
      trajectory.transitions.clear();
      ++pos;
    }
  }

private:
  std::mutex mutex_;
  std::list<trajectory<A, S>> trajectories_;
};

template <typename A, typename S> class policy {
public:
  virtual ~policy() = default;

  virtual A react(const S &state) const = 0;
};

template <std::size_t N, typename S>
class random_policy : public policy<discrete_action<N>, S> {
public:
  virtual discrete_action<N> react(const S &state) const {
    vector v({N});
    v = 1.0 / N;
    discrete_action<N> a;
    a.from_vector(v);
    return a;
  }
};

template <typename A, typename S> class agent {
public:
  explicit agent(const policy<A, S> &p, environment<A, S> &env,
                 replay_buffer<A, S> &rb, std::size_t id = 0)
      : policy_(p), env_(env), replay_buffer_(rb), id_(id) {}
  virtual ~agent() = default;

  // Return whether an episode is open after the step.
  bool step() {
    if (!curr_traj_) {
      // We don't have a history. This is the very first state.
      curr_traj_ = &replay_buffer_.emplace_trajectory(env_.view(id_));
    }

    // There is a past state.
    const S &previous_state = curr_traj_->last_state();
    A action = policy_.react(previous_state);
    env_.apply(action, id_);

    S curr_state = env_.view(id_);
    curr_traj_->add_transition(std::move(action),
                               get_reward(previous_state, curr_state),
                               std::move(curr_state));

    if (game_over(curr_traj_->last_state())) {
      env_.reset(id_);
      curr_traj_->freeze();
      curr_traj_ = nullptr;
      return false;
    }

    return true;
  }

  void play_one_episode() {
    while (step())
      ;
  }

  void play_steps(std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      step();
    }
  }

  std::size_t id() { return id_; }

protected:
  virtual bool game_over(const S &state) = 0;
  virtual float get_reward(const S &state1, const S &state2) = 0;

  std::size_t id_;
  const policy<A, S> &policy_;
  environment<A, S> &env_;
  replay_buffer<A, S> &replay_buffer_;

  trajectory<A, S> *curr_traj_ = nullptr;
};

template <typename A, typename S> class learner {
public:
  explicit learner(replay_buffer<A, S> &rb, model &policy_model,
                   optimizer &policy_optimizer, float gamma = 0.99)
      : replay_buffer_(rb), policy_model_(policy_model),
        policy_optimizer_(policy_optimizer), gamma_(gamma) {}

  void step() { learn(); }

  virtual void learn() = 0;

protected:
  replay_buffer<A, S> &replay_buffer_;
  model &policy_model_;
  optimizer &policy_optimizer_;
  float gamma_;
};

} // namespace xylo

#endif // XYLO_RL_
