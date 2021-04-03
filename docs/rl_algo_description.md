# Deep Reinforcement Learning Introduction
Without lengthy math formulas, we'll try to give a short and not-so-rigorous
introduction for readers new to reinforcement learning. Experienced
RL(reinforcement learning) users should skip to the design. For the detailed
formula-rich review of the algorithms here I strongly recommend Lilian Weng's
[policy gradient
blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html).

## Problem Formulation
This Wikipedia figure captures the reinforcement learning setup.

![RF diagram](Reinforcement_learning_diagram.svg)

There is an **agent** that interacts with a **environment**. The agent perceives
the environment's Markovian **states** and gives **actions** back to it. Each
time an agent makes an action, the environment state will likely change, and the
agent might receive a **reward** associated with the action. The agent makes a
number of steps collecting the state, makes an action and receive a reward. The
sequence of these steps is a **trajectory**. The span of the trajectory is a
**horizon**.


The basic objective of the learning is to maximize the rewards collected along
the entire horizon. For episodic problems trajectories will reach a terminal
state and therefore they are finite. For infinite horizons usually future
rewards are discounted by a factor close to 1, so that the total rewards will be
summed up to a finite number.

Our case study, a variant of bin packing, involves an agent looking at states,
comprising a number of bins and the next item to pack. Its action is the
choosing of a bin for an item. This problem, is a finite-horizon problem,
because the bin capacities are limited, and the bins will eventually be full.
When the bins are full, the packing episode will be over, and we aim to optimize
the total number of items that can be held in the bins given.

## Algorithms Implemented

Popular deep RL algorithms are roughly in 2 camps: **policy gradient** and
**Q-learning**.

* Policy gradient: A policy defines how an agent reacts to environment states.
  Policy gradient methods always have a model that takes in the state and spits
  out an action probability distribution. Through trial-and-error, these methods
  use gradient-ascent like methods to improve the policy over time so that
  better and better policies will be produced.

* Q-learning: Q-learning fits a function that estimates the optimal values of
  state-action pairs. The value of a state-action pair means the expected future
  rewards given a state and an action (or reaction) taken. This function is
  called Q-function. A Q-learning agent takes this value model and uses it to
  recommend the best actions without directly using a policy model.

In this project we have implemented 3 policy gradient methods so far. Q-learning
methods are under investigation, because so far they are not producing better
results than PPO in our case study.

### REINFOCE
REINFOCE is the vanilla policy gradient algorithm. In its simplest form, it has
only one network. Its objective is the future rewards times the log probability
of the action taken given a state. I.e.

<img src="https://render.githubusercontent.com/render/math?math={\nabla_\theta J(\theta)} = {\nabla_\theta \log(\pi_\theta(a | s)) \cdot R^\pi}">

Here <img src="https://render.githubusercontent.com/render/math?math=\theta"> is
the network parameter. _R_ is the total future reward. And
<img src="https://render.githubusercontent.com/render/math?math=\pi"> is the
policy (action probability distribution given a state).  Through
trial-and-error, we'll regress onto this objective and improve our policy.

Using gradient ascent on this objective means that we'll make good actions more
likely over time, and bad actions less likely. Take chess for example, the
future reward will be either 1(win) or -1(lose) at the end of a game.
Through trials, in the future we'll make all moves in the won games more likely,
and all moves in the lost games less likely. For non-binary results, the more
rewards, the more likely the actions will be taken in the future.

In actual implementations rarely raw rewards are used, because there is too much
variance. Advantages are used instead. Advantage means the difference between
the future reward through the action taken, and the state value (expected rewards
given a state). This is equivalent to the original objective because the state
value isn't dependent on the network parameters, and the gradient will be
exactly the same.

### Actor-Critic
REINFOCE can be applied to a wide range of problems, because it doesn't really
assume Markovian states. However, training REINFOCE is slow, because in our
trials we need to wait for the episodes to be over to be able to make one step
of gradient ascent.  Actor-critic introduces one more network (critic network)
that estimates state values, so that we won't have to wait for a terminal state
to be able to estimate advantages.

Our actor critic implementation incorporates [GAE (general advantage
estimation)](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/),
which generally helps learning performance.  This concept is hard to explain.
We'll skip it here.

### PPO (Proximial Policy Optimization)
Policy gradient methods in general suffers from a convergence problem, because
the regression onto the objective isn't really "regression" in the usual sense.
The objective treats rewards as if they are not dependent on the network
parameters, but they actually are a by-product of our policy. When you change
the policy, the reward landscape might become dramatically different, and
therefore we're regressing onto a moving target.

To solve the problem, attempts have been made to limit the policy change in each
gradient ascent step so that bit by bit we're moving towards a better and better
policy while keeping the rewards similar between steps.

PPO is one of these methods. It has two flavors. The first clips update
gradients when the policy changes too much. The second incurs a KL-divergence
regularization loss term to penalize policies from differing. In this project we
implement both.

PPO can be applied to either REINFORCE or actor-critic. The original [PPO
paper](https://arxiv.org/abs/1707.06347) was proposed with a actor-critic
architecture. Therefore we base our PPO algorithm on the actor-critic
implementation as well.

# Design

We'll describe the components in designing the implementation of these
algorithms. Because only policy gradient methods are involved thus far, the
description here is very much policy gradient oriented.

## Environment and Agent
These are the two basic entities interacting with each other in the basic RL
setup and they are represented as abstract classes.

In the code [//xylo//rl.h](xylo/rl.h) describes the interfaces. The environment
is responsible for methods of viewing the current state, and mutating the state
given a certain action. The methods also takes in an agent ID, because the view
to different agents might be slightly different, and actions from different
agents might affect the environment in distinct ways.

Agent is responsible for interpreting whether a state is terminal. In other
frameworks this information might be provided by the environment. It also
interprets the reward given the old and new states.

To implement new applications these classes will need to be overriden to reflect
desired behaviors.

## Algorithms
These algorithms are implemented in learner classes. Each learner represents a
particular kind of algorithm. Usually it takes in a replay buffer, one or more
models, and the optimizer(s) to the model(s). Discount factors are optional.

Replay buffer is a Q-learning concept. It is essentially a cache holding past
agent experiences. Policy gradient methods usually don't involve "replay buffer"
in its most basic sense, but TF-Agents uses a unified replay buffer to record
agents' trajectories, which we have borrowed. We might remove the name "replay
buffer" for policy gradient methods at some point in the future, because the
policy gradient experience replay significantly differs from the Q-learning
"replay buffer", which just returns state transitions, instead of whole
trajectories. 

## State and Action
Environment, agent and learner all take in state and action as template
parameters. Currently only discrete actions are supported. To introduce new
actions, all methods of `xylo::discrete_action` will need to be implemented. The
discrete action can be reused for different applications. The applications are
responsible for properly interpreting action instantiations in the environment
class.

The state class is usually a copy of the environment class's internal states,
since it is the state of the environment after all. The most essential method it
needs to provide is `to_vector`, which turns itself into a tensor.

# Example
Look into [//apps/bin_packing/bin_packing.h](../apps/bin_packing/bin_packing.h)
for how the bin packing study uses these interfaces.
