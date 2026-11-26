# Changelog

## 0.1.9

`hierarchical_start` accepts two ordered `GroupFactor` declarations with a
`GroupedDensity` for crossed or nested normal random effects. The first
factor is integrated by level, then the second is reconstructed using the
curvature of its marginal. Population fits, chain positions and a diagonal
inverse mass are returned through the existing interface.

## 0.1.8

Added `hierarchical_start` for models with population-level parameters and a few
effects per group. It fits the population-level parameters on their marginal
distribution, updates each chain's population values on that marginal by slice
sampling, draws each group's effects from their conditional distribution, and
returns starting positions with a diagonal inverse mass for `Jobs`.

## 0.1.7

Streaming runs can admit children from completed parent draws or saved warmup
transition endpoints. Children inherit frozen tuning, record their ancestry and use
independent random keys while the original chains keep advancing.

## 0.1.6

Added `lower_sample_batches` to inspect the compiler plans for stream startup and
the first round loop before running a chain.

## 0.1.5

`sample_batches` accepts an optional warmup callback for per-chain window closures,
divergences, freezes and failures at their exact rounds. Adaptive diagonal windows
report the stopping statistic and can include the moments used to compute it.

## 0.1.4

Added compact per-chain factors for fixed sampling and windowed warmup. Structured
warmup selects between supplied, diagonal, and learned Householder frames.

## 0.1.3

Added bounded streaming of completed draws with exact parallel-round timestamps.
Each chain can freeze its adaptive warmup and start contributing draws independently.
The stream reports jobs that exhaust their adaptation windows and continues with the others.

## 0.1.2

Warmup now supports independent choices of trajectory evidence, diagonal estimator,
initialization, memory, update frequency, and step-size acceptance statistic.
Fixed and adaptive window schedules can count gradients or transitions per chain.

## 0.1.1

Added optional gradient-batched warmup for a diagonal inverse mass and step size.
The result includes ready-to-sample jobs and per-chain costs for each evaluation window.
