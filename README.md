# Dependency Free Reinforcement Learning
This is a C++ project that implements common algorithms of deep Reinforcement
Learning without introducing any dependencies. Currently a few policy gradient
methods are implemented. This project applies to framework for a bin packing
benchmark for testing and validation. TODO: link a doc to the bin packing app.

# Motivation for Dependency-Free
*Disclaimer:* This project is not related with AlphaZero, NNUE, Stockfish,
Lichess or any external projects mentioned in this section in any way.

Modern deep learning frameworks such as TensorFlow or PyTorch are powerful but
cumbersome. This project aims to provide a lightweight implementation of common
reinforcement learning algorithms without heavy dependencies, to enable more
environments to adopt reinforcement learning. Also, when models are small, the
training time would be significantly less than pure Python implementations such
as [TF-Agents](https://github.com/tensorflow/agents).

An example of a good usecase would be projects similar to
[NNUE](https://www.chessprogramming.org/Stockfish_NNUE). The chessbot community
borrowed ideas from Alpha Zero and introduced a _efficiently updated neural
network_ or _NNUE_ into the state-of-the-art chessbot Stockfish. However,
different from Alpha Zero, which uses GPU-trained heavy models to guide their
Monte-Carlo search, NNUE only adopts a very lightweight model for online
position evaluation in their minimax [Alpha-beta
search](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning).
[Lichess](https://lichess.org/), a prominent figure in online chess,
incorporated NNUE in their own in-browser game analysis. This would require
compiling neural network models into [WebAssembly](https://webassembly.org/) for
best performance.

Under similar circumstances, it'll be very managable to support compiling into
WebAssembly, since our algorithms are written using C++ without depedencies, and
currently compiled with [Clang](https://clang.llvm.org/). For in-browser use,
the major work would be to migrate from Clang to the Clang-based
[Emscripten](https://emscripten.org/) toolchain.

Bringing up this example doesn't imply we'll confine ourselves into in-browser
reinforcement learning. The use case scenarios are limitless. Any project that
is too monolithic to be integrated with a heavy external library will be able to
benefit, as long as the code is compiled through a standard C++ compiler.

# How Dependency-Free are we?
While the project strives to break every dependency there is, pragmatically we
have to make compromises.

## Clang
The project is written in C++20. The current support of C++20 in major compilers
[is progressively being rolled
out](https://en.cppreference.com/w/cpp/compiler_support). We pick Clang as our
compiler due to their better support of the C++20 features used in the project.
That means that we have a hard dependency on Clang now, which will gradually
change as C++20 support slowly rolls out.

## Python and Shell
We write our own build system, instead of using large build systems like
[CMake](https://cmake.org/), which means that we're not subject to their
limitations. However, in order to build our build system, we require that the
environment has Python3 intalled with YAML support. Currently we also assume we
can set environment variables for executing our build scripts and executables.
We currently depend on a Bash script for that.

## File I/O
Since the C++ standard doesn't provide a native I/O API, we do write our own I/O
library, located at [//xeno/sys](xeno/sys) in the source tree. This particular
directory assumes a UNIX-like environment. If I/O is ever required (such as
loading weights from a file) then we would be assuming a POSIX compliant system
interface.

# Installing Dependencies
Currently all development and testing are done on a Ubuntu20.04 box. In this
environment, use the following command to install Clang:
```
sudo apt install clang
```

Python3 can be installed through:
```
sudo apt install python3
sudo apt install python3-pip
pip3 install pyyaml
```

Bash should come by default for Ubuntu.

[//]: # (MathJax test)
[//]: # (<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">)

[//]: # (<img src="https://render.githubusercontent.com/render/math?math=x = {-b \pm \sqrt{b^2-4ac} \over 2a}">)

# Look Around
Aside from the build system, there are 3 major directories of the code. _Xeno_
is a general purpose C++ that covers utilities like logging and string
manipulation and so on. _Xylo_ is a numerical library that includes basic
tensor/matrix operations, neural network layers, and some reinforcement learning
algorithms. _Apps_ currently contains one reinforcement learning application,
bin packing in the context of cluster scheduling.

## Reinforcement Learning
We have currently implemented 3 policy gradient algorithms:
* REINFORCE (or simply policy gradient)
* Online actor-critic
* PPO (proximial policy optimization)

These implementations are all located in
[//xylo/policy_gradient.h](xylo/policy_gradient.h). There is also a
[//xylo/rl.h](xylo/rl.h) that provides more general base classes for both policy
gradient and value-based algorithms such as DQN. Implementing value-based
methods will be our future investigation, as we feel state-of-the-art policy
gradient methods tend to outperform DQN.

Look into [docs/rl_algo_description.md](docs/rl_algo_description.md) for
the design and the code structure of the implementations.

## Bin Packing
As we don't have access to the OpenAI gym, we are testing out our algorithms on
an interesting application -- bin packing. [Bin
packing](https://en.wikipedia.org/wiki/Bin_packing_problem) itself is an
interesting NP-hard problem. In the cluster scheduling context, people often
run into even more challenging variants, such as multi-dimensional packing. We
found the study of the bin packing problem itself is sophisticated enough, and
we are currently expanding it into a research paper.  Look in
[docs/binpacking.md](docs/binpacking.md) for the study. You can go into
[//apps/bin_packing](apps/bin_packing) to play with some training code while
reading the doc. The binpacking document will guide you through the build and
command execution.
