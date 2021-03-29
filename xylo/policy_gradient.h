#include "xylo/tensor.h"
#include <xylo/rl.h>

namespace xylo {

namespace {
template <typename A, typename S>
inline std::size_t num_transitions(const std::vector<td<A, S>> &experience) {
  std::size_t result = 0;
  for (const auto &traj : experience) {
    result += traj.size();
  }
  return result;
}

template <typename A>
inline matrix policy_loss(const std::vector<A> &actions, vector_view advantages,
                          matrix_view orig_action_matrix) {
  matrix result({actions.size(), A::cardinality()});
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const auto &action = actions[i];
    action.softmax_gradient_log(orig_action_matrix[i], result[i],
                                advantages[i]);
  }
  return result;
}

template <typename A>
inline matrix surrogate_loss(const std::vector<A> &actions,
                             vector_view advantages,
                             matrix_view orig_action_matrix) {
  matrix result({actions.size(), A::cardinality()});
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const auto &action = actions[i];
    action.clipped_gradient(orig_action_matrix[i], result[i], advantages[i]);
  }
  return result;
}

// D_KL(P || Q)
inline float kl_divergence(vector_view p, vector_view q) {
  if (p.size() != q.size())
    throw std::exception();
  return sum(p * ::log(p / q));
}

template <typename A>
inline matrix kl_regulated_loss(const std::vector<A> &actions,
                                vector_view advantages, float d_targ,
                                float &beta, matrix_view orig_action_matrix) {
  matrix result({actions.size(), A::cardinality()});
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const auto &action = actions[i];
    action.softmax_gradient_log(orig_action_matrix[i], result[i],
                                advantages[i]);
  }

  int i = 0;
  matrix action_matrix({actions.size(), A::cardinality()});
  for (vector_view v : matrix_view(action_matrix)) {
    v = *(actions[i++].distrib);
  }
  matrix regulation =
      softmax_cross_entropy_loss_grad(action_matrix, orig_action_matrix);
  flatten(regulation) *= beta;

  result += regulation;

  float d_average = 0;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    float d = kl_divergence(action_matrix[i], orig_action_matrix[i]);
    d_average += d;
  }
  d_average /= actions.size();
  // lg() << "d: " << d;
  if (std::abs(d_average) < d_targ / 1.5) {
    beta /= 2;
  } else if (std::abs(d_average) > d_targ * 1.5) {
    beta *= 2;
  }
  beta = std::max<float>(beta, 1e-25);
  beta = std::min<float>(beta, 0.1);
  // lg() << "beta: " << beta;
  return result;
}

} // namespace

template <typename A, typename S>
class policy_gradient_learner : public learner<A, S> {
public:
  policy_gradient_learner(replay_buffer<A, S> &rb, model &action_model,
                          optimizer &action_optimizer, float gamma = 1)
      : learner<A, S>(rb, action_model, action_optimizer, gamma) {}
  virtual void learn() override {
    // Takes stuff from the replay buffer, calculate advantages and take a step
    // toward a better action policy.

    std::vector<td<A, S>> experience =
        learner<A, S>::replay_buffer_.sample_td();

    std::size_t total_num_transitions = num_transitions(experience);
    std::size_t state_length = S::length();

    matrix state_matrix({total_num_transitions, state_length});

    std::size_t curr = 0;
    std::vector<A> actions;
    for (const auto &traj : experience) {
      const std::size_t traj_size = traj.size();
      for (const auto &transition : traj) {
        transition.start_state->to_vector(state_matrix[curr]);
        actions.push_back(transition.action);
        ++curr;
      }
    }

    vector advantages = get_advantages(experience);
    this->policy_optimizer_.step(state_matrix,
                                 [&](xylo::matrix_view v) -> matrix {
                                   return policy_loss(actions, advantages, v);
                                 });
  }

  vector get_advantages(const std::vector<td<A, S>> &experience) {
    vector rewards_to_go({num_transitions(experience)});
    const float discount = this->gamma_;
    float total_reward = 0;

    std::size_t curr = 0;
    for (const auto &traj : experience) {
      vector_view reward_slice = slice(rewards_to_go, curr, traj.size());
      auto traj_pos = traj.begin();
      auto reward_pos = reward_slice.end() - 1;
      float reward = 0;

      for (;
           traj_pos != traj.end() && reward_pos != reward_slice.begin() - 1;) {
        reward = (traj_pos++)->reward + discount * reward;
        *reward_pos-- = reward;
      }
      total_reward += reward_slice[0];
      curr += traj.size();
    }
    float avg_reward = total_reward / experience.size();
    return rewards_to_go - avg_reward;
  }
};

template <typename A, typename S>
class actor_critic_learner : public learner<A, S> {
public:
  actor_critic_learner(replay_buffer<A, S> &rb, model &action_model,
                       optimizer &action_optimizer, model &value_model,
                       optimizer &value_optimizer, float gamma = 0.99)
      : learner<A, S>(rb, action_model, action_optimizer, gamma),
        value_model_(value_model), value_optimizer_(value_optimizer) {}

  virtual void learn() override {
    // TODO: fill out this part.
    std::vector<td<A, S>> experience =
        learner<A, S>::replay_buffer_.sample_td();

    std::size_t state_length = S::length();
    std::size_t total_num_transitions = num_transitions(experience);

    // Adding a few end states.
    matrix state_matrix(
        {total_num_transitions + experience.size(), state_length});
    std::vector<A> actions;

    std::size_t curr = 0;
    for (const auto &traj : experience) {
      for (const auto &transition : traj) {
        transition.start_state->to_vector(state_matrix[curr++]);
        actions.push_back(transition.action);
      }
      actions.push_back(actions.back());
      traj.back().end_state.to_vector(state_matrix[curr++]);
    }

    update_value_model(experience, state_matrix);
    vector advantage = calculate_advantage(experience, state_matrix);
    optimize_action(state_matrix, actions, advantage);
  }

  virtual void optimize_action(matrix_view state_matrix,
                               const std::vector<A> &actions,
                               vector_view advantage) {
    this->policy_optimizer_.step(state_matrix,
                                 [&](xylo::matrix_view v) -> matrix {
                                   return policy_loss(actions, advantage, v);
                                 });
  }

  void update_value_model(const std::vector<td<A, S>> &experience,
                          matrix_view state_matrix) {
    // TODO: Reimplement after the great refactor.
    std::size_t state_length = S::length();
    matrix value_matrix = value_model_.eval(state_matrix);
    vector_view values = flatten(value_matrix);

    // lg() << "values: " << xeno::string::streamable(values);

    std::size_t curr = 0;
    vector updated_values({values.size()});
    for (const auto &traj : experience) {
      for (const auto &transition : traj) {
        updated_values[curr] =
            transition.reward + learner<A, S>::gamma_ * values[curr + 1];
        ++curr;
      }
      updated_values[curr] = values[curr];
      ++curr;
    }
    value_optimizer_.step(state_matrix,
                          std::bind_front(square_loss_grad, updated_values));
  }

  vector calculate_advantage(const std::vector<td<A, S>> &experience,
                             matrix_view state_matrix) {
    std::size_t state_length = S::length();
    matrix value_matrix = value_model_.eval(state_matrix);
    vector_view values = flatten(value_matrix);
    vector advantage({values.size()});
    vector deltas({values.size()});

    // This is an optimization. We know for a fact at the end of the episode the
    // reward is 0. Without this the algorithm still converges, but more slowly.
    std::size_t curr = 0;
    for (const auto &traj : experience) {
      curr += traj.size();
      if (traj.frozen())
        values[curr] = 0;
      ++curr;
    }

    curr = 0;
    for (const auto &traj : experience) {
      for (auto &transition : traj) {
        deltas[curr] = transition.reward +
                       learner<A, S>::gamma_ * lambda_ * values[curr + 1] -
                       values[curr];
        ++curr;
      }
      // End state of the trajectory.
      advantage[curr] = 0;
      ++curr;
    }

    curr = 0;
    std::size_t episode_index = 0;
    for (const auto &traj : experience) {
      for (auto &transition : traj) {
        deltas[curr] = transition.reward +
                       learner<A, S>::gamma_ * values[curr + 1] - values[curr];
        ++curr;
      }
      // End state of the trajectory.
      deltas[curr] = 0;
      ++curr;
    }

    curr = 0;
    for (const auto &traj : experience) {
      std::size_t traj_end = curr + traj.size();
      for (auto &transition : traj) {
        advantage[curr] = 0;
        float coefficient = 1;
        for (std::size_t i = curr; i < traj_end; ++i) {
          advantage[curr] += deltas[i] * coefficient;
          coefficient *= lambda_ * learner<A, S>::gamma_;
        }
        ++curr;
      }
      // End state of the trajectory.
      advantage[curr] = 0;
      ++curr;
    }
    return advantage;
  }

protected:
  model &value_model_;
  optimizer &value_optimizer_;
  float lambda_ = 0.95;
};

template <typename A, typename S>
class ppo_learner : public actor_critic_learner<A, S> {
public:
  ppo_learner(replay_buffer<A, S> &rb, model &action_model,
              optimizer &action_optimizer, model &value_model,
              optimizer &value_optimizer, float gamma = 0.99)
      : actor_critic_learner<A, S>(rb, action_model, action_optimizer,
                                   value_model, value_optimizer, gamma) {}
  virtual void optimize_action(matrix_view state_matrix,
                               const std::vector<A> &actions,
                               vector_view advantage) {
    constexpr std::size_t k = 4;
    for (std::size_t i = 0; i < k; ++i) {
      this->policy_optimizer_.step(
          state_matrix, [&](xylo::matrix_view v) -> matrix {
            return surrogate_loss(actions, advantage, v);
          });
    }
  }
};

template <typename A, typename S>
class kl_ppo_learner : public actor_critic_learner<A, S> {
public:
  kl_ppo_learner(replay_buffer<A, S> &rb, model &action_model,
                 optimizer &action_optimizer, model &value_model,
                 optimizer &value_optimizer, float gamma = 0.99)
      : actor_critic_learner<A, S>(rb, action_model, action_optimizer,
                                   value_model, value_optimizer, gamma) {}
  virtual void optimize_action(matrix_view state_matrix,
                               const std::vector<A> &actions,
                               vector_view advantage) {
    constexpr std::size_t k = 4;
    for (std::size_t i = 0; i < k; ++i) {
      this->policy_optimizer_.step(
          state_matrix, [&](xylo::matrix_view v) -> matrix {
            matrix loss =
                kl_regulated_loss(actions, advantage, d_targ_, beta_, v);
            return loss;
          });
    }
  }

private:
  float beta_ = 1;
  float d_targ_ = 1e-9;
};

template <typename A, typename S>
class policy_gradient_policy : public policy<A, S> {
public:
  policy_gradient_policy(model &m) : m_(m) {}

protected:
  A react(const S &state) const override {
    vector state_vector = to_vector(state);
    matrix action_vector =
        m_.eval(fold<2>(state_vector, {1, state_vector.size()}));
    A action;
    action.from_vector(flatten(action_vector));
    return action;
  }

private:
  const model &m_;
};

template <typename A, typename S>
class policy_gradient_deterministic_policy : public policy<A, S> {
public:
  policy_gradient_deterministic_policy(model &m) : m_(m) {}

protected:
  A react(const S &state) const override {
    vector state_vector = to_vector(state);
    matrix action_vector =
        m_.eval(fold<2>(state_vector, {1, state_vector.size()}));
    A action;
    action.from_vector_deterministic(flatten(action_vector));
    return action;
  }

private:
  const model &m_;
};
} // namespace xylo
