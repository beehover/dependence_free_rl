# Dependency Free Reinforcement Learning
This is a C++ project that implements common algorithms of deep Reinforcement
Learning without introducing any dependencies. Currently a few policy gradient
methods are implemented. This project applies to framework for a bin packing
benchmark for testing and validation. TODO: link a doc to the bin packing app.

# Motivation for Dependency-Free
Modern deep learning frameworks such as TensorFlow or PyTorch are powerful but
cumbersome. This project aims to provide a lightweight implementation of common
reinforcement learning algorithms without heavy dependencies, to enable more
environments to adopt reinforcement learning.

An example of such an usecase is
[NNUE](https://www.chessprogramming.org/Stockfish_NNUE). The chessbot community
borrowed ideas from AlphaGo and introduced a lightweight neural network model
into their minimax [Alpha-beta
pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning).
[Lichess](https://lichess.org/), a prominent figure in online chess,
incorporated
NNUE in there own in-browser game analysis. This would require compiling neural
network models into [WebAssembly](https://webassembly.org/) for best performance.

While we are disclaiming that this project is not related to NNUE or Lichess in
any way, since our models are written using C++ and currently compiled with
[Clang](https://clang.llvm.org/), it'll be trivial to extend to support
compiling into WebAssembly through the [emscripten](https://emscripten.org/) tool chain for in-browser use.

Bringing up this example doesn't imply we'll confine ourselves into in-browser
reinforcement learning. The use case scenarios are limitless. Any project that
is too monolithic to be integrated with a heavy external library will be able to
benefit, as long as the code is compiled through a standard C++ compiler.

