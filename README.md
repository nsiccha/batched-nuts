# batched-nuts

The No-U-Turn Sampler (NUTS) for many chains at once in [JAX](https://github.com/jax-ml/jax),
batched by gradient evaluation instead of by transition.

## The problem

The usual way to run many NUTS chains on one device is `jax.vmap` over a NUTS kernel,
for example `blackjax.nuts`. The chains then move in step: a transition of the batch is
over when its *longest* trajectory is. NUTS chooses the length of every trajectory on the
fly, and the lengths differ by powers of two from chain to chain and from one transition
to the next. So in most transitions most chains have finished early and wait, and the
device keeps evaluating gradients for them that are thrown away.

```
by transition (vmapped kernel)            by gradient evaluation (this package)

chain 1  ███·····|█·······|               lane 1  ███|█|███████|██|█████…
chain 2  ████████|███·····|               lane 2  ████████|███|████|███…
chain 3  █·······|████████|               lane 3  █|████████|██|██████…

█ a leapfrog step of the chain    · a gradient that is evaluated and discarded    | a draw
```

## What this package does

The chains run in the *lanes* of one batch. Every iteration of the batched loop makes
exactly one leapfrog step, that is one gradient evaluation, in every lane. What the step
means differs from lane to lane: it may extend a trajectory, complete a subtree, or end a
transition, in which case the lane records its draw and starts its next transition in the
following iteration. No lane waits for another.

There can be more chains than lanes. The chains that are not running wait in a *queue*,
and a lane whose chain has all its draws takes the next chain from it. That chain may
belong to a different posterior: every chain names the slice of the data its log density
reads. This is what keeps a device busy in a simulation study with thousands of small
posteriors.

The transition itself is the one of `blackjax.nuts` (version 1.7.1, with its defaults:
velocity Verlet, at most 10 doublings, divergence at an energy error of 1000), for a
diagonal mass matrix. Given the same random keys it builds the same trajectories and
returns the same draws; [what the tests show](#what-the-tests-show) says exactly how
far "the same" goes.

## Usage

Python 3.11 or later, JAX 0.12 or later. From a clone of this repository:

```
pip install .            # the package: needs only jax and numpy
pip install ".[test]"    # also pytest, blackjax and scipy, for the tests and the benchmark
```

```python
import jax
import jax.numpy as jnp

import batched_nuts

jax.config.update("jax_enable_x64", True)


def logdensity(position, data):
    """The target of one chain: a normal distribution, described by one slice of `data`."""
    return -0.5 * jnp.sum(((position - data["mean"]) / data["scale"]) ** 2)


# 50 targets ("datasets") in 4 dimensions.  Their scales are more and more spread out,
# so their chains need longer and longer trajectories.
num_datasets, num_chains, dim = 50, 4, 4
key, mean_key, start_key = jax.random.split(jax.random.key(0), 3)
spread = jnp.linspace(0.0, 3.0, num_datasets)[:, None]
data = {
    "mean": jax.random.normal(mean_key, (num_datasets, dim)),
    "scale": jnp.exp(-spread * jnp.linspace(0.0, 1.0, dim)),
}

# 4 chains per dataset: 200 chains, one row each.
dataset = jnp.repeat(jnp.arange(num_datasets), num_chains)
noise = jax.random.normal(start_key, (len(dataset), dim))
jobs = batched_nuts.Jobs(
    key=jax.random.split(key, len(dataset)),
    position=data["mean"][dataset] + data["scale"][dataset] * noise,
    step_size=0.5 * data["scale"][dataset].min(axis=1),
    inverse_mass=jnp.ones((len(dataset), dim)),
    dataset=dataset,
)

# 32 chains run at a time; the other 168 wait in the queue.
draws = batched_nuts.sample(logdensity, jobs, num_draws=250, data=data, num_lanes=32)

print(draws.position.shape)  # (200, 250, 4): chains, draws, dimensions
print(draws.num_integration_steps.mean(axis=1))  # from 7 to 50 leapfrog steps per draw
useful = draws.num_evaluations / (draws.num_iterations * 32)
print(f"{useful:.0%} of the gradient evaluations advanced a chain")
```

`sample` returns NumPy arrays with the leading axes (chains, draws): `position`,
`logdensity`, and per draw `num_integration_steps`, `is_divergent`, `is_turning` and
`acceptance_rate`, named as in BlackJAX. `num_iterations` counts the iterations of the
batched loop, each of which cost one gradient evaluation per lane, and `num_evaluations`
counts the gradient evaluations that advanced a chain.

A log density without data takes the position only: `sample(logdensity, jobs, num_draws)`.
Without `num_lanes`, all chains run at once.

## Optional diagonal warmup

Give each chain an initial step size and positive diagonal inverse mass in `jobs`.
Warmup returns the last positions, frozen parameters and fresh sampling keys together
as `fit.jobs`:

```python
fit = batched_nuts.warmup(logdensity, jobs, data=data, num_lanes=32)
draws = batched_nuts.sample(logdensity, fit.jobs, num_draws=250, data=data, num_lanes=32)
print(fit.num_evaluations[0])  # actual leapfrog count in each window of chain 0
```

With the default settings each chain uses its own gradient-evaluation windows. The default
`window_evaluations=(1000, 2000, 1000)` gives per-window leapfrog budgets.
A chain finishes the trajectory that crosses its budget before starting the next
window. `num_evaluations`, `num_transitions` and `num_divergences` report the
actual counts for each chain and window.

In this configuration the sampler accumulates position and log-density gradient
moments with the probabilities used to select the NUTS proposal. Each completed
trajectory contributes one unit of weight to its window, including its within-trajectory
variance. The diagonal inverse mass estimate is the square root of the ratio of the
two coordinatewise variances. A short window shrinks the log estimate toward the
previous diagonal with a weight of three transitions. Each metric window installs its estimate
before the next momentum draw. The final window tunes only the step size.

Step-size adaptation uses dual averaging on the mean leapfrog acceptance, with
`target_acceptance=0.8` by default. Each metric refit restarts dual averaging.
The final smoothed step size and metric stay fixed during sampling.

### Choosing how the metric learns

For immediate updates with NUTS weights and a per-chain stopping rule:

```python
diagonal = batched_nuts.DiagonalAdaptation(
    evidence="nuts", estimator="position_score", initial="gradient",
    memory="overlapping", switch_interval=80, switch_clock="transitions",
    update_interval=1, shrinkage=0,
)
schedule = batched_nuts.AdaptiveWindows(
    first_window=1000, growth=2, clock="gradients", tolerance=2,
    max_windows=10, final_window=45, final_clock="transitions",
)
fit = batched_nuts.warmup(
    logdensity, jobs, data=data, num_lanes=32, diagonal=diagonal,
    schedule=schedule, acceptance="late_symmetric", restart_dual=False,
)
```

| Choice | What it controls |
| --- | --- |
| `evidence` | `"selected"` uses one draw per transition, including repeats. `"eligible_selected"` skips the starting point and, on a divergent transition, requires a draw more than four steps from the start. `"nuts"` gives each trajectory unit weight and integrates over its proposal probabilities. `"eligible_leaves"` gives each finite noninitial leapfrog point unit weight when its energy increase is below `log(100)`, including leaves of discarded subtrees. |
| `estimator` | `"position"` fits coordinate variances. `"position_score"` fits the square root of position variance divided by gradient variance. `"fixed"` keeps the initial diagonal and tunes the step size. |
| `initial`, `include_initial` | `"provided"` starts from the inverse mass in `jobs`. `"gradient"` uses the clipped reciprocal magnitude of the starting gradient and uses the supplied entry for zero gradients. Set `include_initial=True` to seed the fit with the starting position and gradient once. |
| `memory`, `switch_interval` | `"window"` clears the observations at each switch. `"cumulative"` keeps them. `"overlapping"` collects a second estimate in parallel, promotes it at a switch and starts a new background estimate. Switch at metric-window boundaries (default) or after an integer count in `switch_clock`, which defaults to the metric-window clock. |
| `update_interval` | Install the estimate at each metric-window boundary (default), or additionally after a positive count of transitions or gradients in `update_clock`. A value of 1 with the default transition clock applies the diagonal to the next trajectory. Updates always wait for a complete trajectory. |
| `variance`, `shrinkage` | `"population"` computes centered, weighted population variance. `"pre_update"` measures deviations from the previous running mean as in nutpie, adding within-trajectory variance for NUTS weights. `shrinkage` is the prior weight on the installed log diagonal at each fit (default 3, zero for a full update). The data weight counts trajectories, or individual eligible leaves for the leaf source. Unidentified coordinates keep their previous values. |
| `schedule` | `FixedWindows(metric_windows=(...), final_window=..., clock=...)` specifies the lengths in `"transitions"` or `"gradients"`. `AdaptiveWindows(...)` tests evidence at growing per-chain window boundaries. Both schedules accept a separate `final_clock` for the step-size-only window. |
| `acceptance`, `restart_dual` | Dual averaging can use mean acceptance, symmetric acceptance over the leapfrog steps, or symmetric acceptance only during final calibration (`"mean"`, `"symmetric"`, `"late_symmetric"`). `restart_dual=True` restarts the step-size controller at each metric-window boundary. |

The adaptive test estimates a fresh diagonal in each of two consecutive, disjoint windows.
For each coordinate it takes the square root of the ratio of the two estimates. Metric
learning ends when all coordinates are identified and the largest ratio divided by the
smallest is below `tolerance`. The comparison allows a common change of scale that the
step size can absorb. It measures stability of the diagonal shape. Posterior accuracy still
requires diagnostics on retained draws. Set `AdaptiveWindows(evidence=...)` to give this
test a different source from the fitted metric. On exhaustion of `max_windows` warmup
raises an error identifying the affected jobs. Unused metric windows have zero counts.
The last column always records final step-size calibration.

## What you provide, and what the package leaves out

- **Step size and mass matrix.** Every chain has a fixed step size and a fixed diagonal
  inverse mass matrix during sampling. Supply them in `Jobs` or use the jobs returned
  by `warmup`. The benchmark uses `blackjax.window_adaptation` for its common start.
- **Positions are vectors.** Flatten a structured position yourself, for example with
  `jax.flatten_util.ravel_pytree`.
- **Keys.** Transition `t` of a chain uses the key `jax.random.fold_in(key, t)` exactly
  as `blackjax.nuts(...).step` would use it.
- **Memory.** The device holds the state of the lanes and a buffer for finished
  transitions (`buffer_size`, by default `max(1024, 4 * num_lanes)` draws). Whenever the
  buffer is full the draws move to NumPy arrays in host memory, so the number of chains
  and draws is not limited by device memory. The queue of waiting chains stays in host
  memory too.
- **Compilation.** The loop is compiled once per log density function and `max_depth`.
  Pass the same function object again to reuse it.

## What the tests show

`tests/test_blackjax.py` runs the same chains with `batched_nuts.sample` and with
`jax.vmap` over `blackjax.nuts`, given the same keys, on three synthetic targets: normal
distributions with scales from 0.1 to 3, Neal's funnel, and logistic regressions on
simulated data. The chains are chosen to behave very differently: trajectories from 1 to
255 leapfrog steps, U-turns, trajectories of the maximal length, and divergences. There
are fewer lanes than chains, so lanes take over new chains and new datasets on the way.

- **The trajectories are identical.** Every one of the 1,000 transitions has the same
  number of leapfrog steps, the same divergence flag and the same U-turn flag.
- **The draws are identical bit for bit** on the normal target, whose gradient is an
  elementwise expression. That test asserts exact equality.
- **On the other two targets the draws agree at least to rounding error**, and the test
  allows a relative difference of 1e-8. The reason is not in the sampler. The compiler
  behind JAX (XLA) may round a sum, or a multiplication followed by an addition,
  differently in two different programs, and gradients that contain sums pick that up
  in their last bit. Such a difference then grows slowly along a chain. On the CPU of
  the machine used for the benchmark below, the draws of these two targets are in fact
  also equal bit for bit. A funnel with 101 dimensions agrees to 5e-11 after 20
  transitions there, and on the GPU the logistic regressions of the benchmark agree to
  9e-15 after 200.

`tests/test_refill.py` checks that the draws of a chain do not depend on the number of
lanes or on the size of the buffer, and `tests/test_distribution.py` checks the moments
of the draws on two normal targets.

```
pytest
```

## Benchmark

`benchmarks/benchmark.py` is a small simulation study: 64 simulated datasets for a
logistic regression (500 rows, 10 predictors), 4 chains each, 200 draws per chain. The
predictors of a dataset are correlated, weakly in some datasets and strongly in others,
so that the chains need between 7 and 114 leapfrog steps per draw on average (32 over
all chains). Every chain is first adapted with `blackjax.window_adaptation`; warm-up is
not timed. Then 64 chains run at a time.

Both samplers do the same useful work, the leapfrog steps of the trajectories. A gradient
evaluation is *useful* if it is one of those steps; the samplers differ in how many
gradient evaluations they pay for them.

| 256 chains, 64 at a time | gradient evaluations | useful | one CPU thread | GPU |
| --- | --- | --- | --- | --- |
| `jax.vmap` of `blackjax.nuts`, in batches of 64 chains | 5.6 million | 29% | 28 s | 8.3 s |
| `batched_nuts.sample`, the same batches of 64 chains | 4.8 million | 34% | 35 s | 8.2 s |
| `batched_nuts.sample`, 64 lanes, the other chains queued | 2.6 million | 64% | 19 s | 4.9 s |

Measured with JAX 0.12 and BlackJAX 1.7.1 in double precision, on one thread of an AMD
Ryzen 9 9950X and on an NVIDIA GeForce RTX 5060 Ti. Compilation is not included. The
counts are those of the CPU run; on the GPU the warm-up ends in slightly different states
and the useful shares are 29%, 35% and 66%. For the vmapped kernel the count is a lower
bound, see the script.

Read as follows.

- **With a queue** the lanes are busy most of the time. They are not busy all of the
  time because the queue runs dry at the end: the last, slowest chains finish alone.
  The longer the queue, the less this matters. With 1,024 chains for the same 64 lanes
  (`--datasets 256`), 95% of the gradient evaluations are useful, against 33% for the
  vmapped kernel, and the run takes 48 s instead of 91 s on the CPU.
- **Without a queue** (the same 64 chains in 64 lanes until all are done) batching by
  gradient evaluation gains little here. The posteriors differ, so a lane with an easy
  posterior finishes early and then idles until the hardest one is done.
- **One iteration costs more than one step of the vmapped kernel**: 0.47 ms against
  0.32 ms on the CPU and 0.11 ms against 0.09 ms on the GPU, for 64 lanes. The loop
  derives random numbers and draws a momentum in every lane in every iteration, whereas
  the vmapped kernel does that once per doubling or once per transition. With a log
  density as cheap as this one, that bookkeeping is a large part of an iteration; the
  more expensive the gradient, the less it matters.

```
python benchmarks/benchmark.py
```

## How it works

The sampling loop is in `src/batched_nuts/nuts.py`. The per-lane adaptation is in `warmup.py`.

- `_step` advances one lane by one leapfrog step and does the bookkeeping of NUTS for
  it without any Python control flow, so that `jax.vmap` turns it into the step of all
  lanes: progressive sampling within the current subtree, the U-turn checks of the
  balanced blocks of that subtree, and, when the subtree is complete, merging it into the
  trajectory and deciding whether the transition ends.
- `_loop` repeats that step for all lanes in a `jax.lax.while_loop` and appends the
  finished transitions to the buffer. It returns to Python when all chains are done,
  when the buffer is full, or when a lane is idle while chains are waiting.
- `sample` is the Python loop around it. It moves finished draws to host memory and
  hands waiting chains to idle lanes (`_refill`).

## Related work

- M. D. Hoffman and A. Gelman, "The No-U-Turn Sampler", JMLR 15, 2014.
- [BlackJAX](https://github.com/blackjax-devs/blackjax), whose NUTS transition this
  package follows, and [NumPyro](https://github.com/pyro-ppl/numpyro), from which the
  iterative form of NUTS and its U-turn checks come.
- A. Radul, B. Patton, D. Maclaurin, M. D. Hoffman and R. A. Saurous, "Automatically
  Batching Control-Intensive Programs for Modern Accelerators", MLSys 2020
  ([arXiv:1910.11141](https://arxiv.org/abs/1910.11141)), solve the same problem for
  NUTS in general: a compiler transformation that tracks a program point for every
  member of a batch. This package does it by hand for one algorithm.

## Pooled draws during a run

Use `sample_batches` to keep every chain sampling until a common deadline or
gradient budget. Each job occupies one lane. With a warmup schedule, a chain can
emit samples as soon as its own metric and step size freeze. With fitted jobs,
omit the schedule. The first completed draw is returned immediately and subsequent
draws are transferred to host memory in bounded batches.

```python
from batched_nuts import AdaptiveWindows, sample_batches

for batch in sample_batches(logdensity, jobs, schedule=AdaptiveWindows(),
                            buffer_size=1024, max_rounds=50_000):
    q = batch.records.position
    chain = batch.records.job
    completed_at = batch.completion_rounds
    # Consume q here. Keeping a running sum and count uses fixed host memory.
```

`max_rounds` counts parallel leapfrog rounds from the start of warmup. The one
batched evaluation of the starting positions precedes round one. Omit the
limit to consume the iterator for as long as needed. `rounds_per_batch` can
bound the intervals between yields, including empty warmup checkpoints.
`buffer_size` must accommodate one simultaneous draw from every lane.
Device storage scales with lane count, parameter dimension, tree-depth limit
and buffer capacity, independently of how long the iterator runs.
If adaptive warmup exhausts its windows for a chain, the batch reports its ID
in `failed_jobs` once. The remaining chains continue sampling. If every chain
exhausts its windows, the stream ends after the last failure report.

## License

MIT
