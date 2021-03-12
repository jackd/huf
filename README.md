# Haiku Utilities and Framework: [huf](https://github.com/jackd/huf)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- Various utilities for working with [haiku](https://github.com/deepmind/dm-haiku)
- A minimal keras-inspired framework for supervised learning
- [gin](https://github.com/google/gin-config) configurable CLI

## Installation

After installing [jax](https://github.com/google/jax)

```bash
pip install dm-haiku
git clone https://github.com/jackd/huf.git
cd huf
pip install -e .
```

## Quick Start

See the [mnist example](examples/mnist/fit.py) for a simple classification example. To use the CLI,

```bash
cd examples/mnist
python -m huf huf_config/fit.gin configs/base.gin
```

You can also experiment with tweaked configurations:

```bash
python -m huf huf_config/fit.gin configs/base.gin --bindings='
batch_size = 32
epochs = 12
dropout_rate = 0.6
'
```

Note this is equivalent to

```bash
python -m huf huf_config/fit.gin configs/tweaked.gin
```

## Projects using HUF

- [grax](https://github.com/jackd/grax): graph networks with jax

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```

## TODO

- Document everything
- Seperate `jax` data library that focuses just on data, like tf.data (`dax`?, `jata`?)
