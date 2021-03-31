# Deep Reinforcement Learning Introduction
We'll try to give a short and not-so-rigorous introduction for readers new to
reinforcement learning. Experienced RL users should skip to the design. For the
detailed review of the algorithms here I strongly recommend Lilian Weng's
[policy gradient
blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html).

## Problem Formulation
This Wikipedia figure captions the reinforcement learning set up.

![RF diagram](https://en.wikipedia.org/wiki/Reinforcement_learning#/media/File:Reinforcement_learning_diagram.svg)

There is an *agent* that interacts with a *environment*. The agent perceives the
environment's Markovian *states* and gives *actions* back to it. Each time an
agent makes an action, the environment state will likely change, and the agent
might receive a *reward* associated with the action. The agent makes a number of
steps collecting the state, makes an action and receive a reward. The sequence
of these steps is a *trajectory*. The span of the trajectory is a *horizon*.


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
