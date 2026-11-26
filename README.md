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

The transition follows `blackjax.nuts` 1.7.1 with velocity Verlet, at most 10 doublings,
and a divergence threshold of 1000. With a diagonal inverse mass and the same random
keys, it reproduces the trajectories and draws described in
[what the tests show](#what-the-tests-show). For a correlated target, each chain can
carry its own [compact metric factor](#correlated-metrics).

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

## Correlated metrics

A job can carry a factor `L` of its inverse mass `L @ L.T`. With a Pathfinder
covariance of the form `diag(a) + B @ C @ B.T`, pass the diagonal variances, basis,
and symmetric core for each chain. The low-rank correction may have either sign.
The resulting covariance must be positive definite.

```python
from batched_nuts import MetricFactor, StructuredAdaptation

# a: (chains, dim), B: (chains, dim, rank), C: (chains, rank, rank)
factor = MetricFactor.from_low_rank(a, B, C)
jobs = jobs._replace(inverse_mass=factor)
policy = StructuredAdaptation(capacity=64, max_reflections=8, score="mean_product")
fit = batched_nuts.warmup(logdensity, jobs, data=data, structured=policy)
draws = batched_nuts.sample(logdensity, fit.jobs, num_draws=100, data=data)
```

`sample` takes the supplied factor directly. In `warmup` or `sample_batches` with
a schedule, a factor starts the first trajectory in the supplied frame and enables
structured adaptation. Passing `structured=policy` with ordinary diagonal jobs
also enables it. The fitted factor is carried through lane refill and frozen for
sampling. Positions and log densities are returned in the target's original coordinates.

At each metric-window boundary the chain fits three frames to the same uniform
reservoir of up to `capacity` completed NUTS proposals and their gradients.
One frame uses the supplied factor, one uses the original axes, and one learns
up to `max_reflections` Householder reflections. The reflection search examines
leading gradient directions among coordinates with large squared log-products.
Each frame fits an additional diagonal scale from position and gradient spread.

For a frame `L`, compute `u` per coordinate as the sample standard deviation of
`solve(L, position)` times that of `L.T @ gradient`. The default score averages
`u` across coordinates. Set `score="squared_log"` to minimize `sum(log(u)**2)`.
A window without a finite candidate score keeps the current metric. An adaptive
schedule permits stopping from its second fitting window when both the spread of
the active frame's diagonal scale corrections and the improvement in frame score
fall below its tolerance. The final window calibrates only the step size.

The retained evidence uses `2 * capacity * dim` floats per lane. The factor
stores two diagonal vectors, the reflection vectors, and a small triangular block.
The thin singular-value decomposition at a window boundary uses a matrix of shape
`(capacity, dim)`. The original diagonal policy keeps its streaming moment estimates.

Run `PYTHONPATH=src python benchmarks/structured_warmup.py` for a matched
correlated-Gaussian comparison of time to first draw, pooled effective sample
size and gradient cost. It uses independent chains to estimate the variance of
the pooled mean along the long axis of the Gaussian.

## Starts for hierarchical models

For a model with population-level parameters and a few effects per group,
`hierarchical_start` computes starting positions from the model's structure. You name
the population-level coordinates and the coordinates of each group's effects:

```python
start = batched_nuts.hierarchical_start(
    logdensity,
    population=population,  # (num_population,) coordinates
    groups=groups,  # (num_groups, effects_per_group) coordinates
    num_chains=128,
    key=jax.random.key(0),
)
jobs = batched_nuts.Jobs(
    key=jax.random.split(jax.random.key(1), 128),
    position=start.position,
    step_size=np.full(128, 0.4),
    inverse_mass=start.inverse_mass,
)
fit = batched_nuts.warmup(logdensity, jobs)
```

The population-level parameters are fitted on their marginal distribution, and each
group's effects are then drawn from their conditional distribution:

- **Conditional effects.** For fixed population values, the effects of every group move
  to their conditional mode by Newton steps on one small block per group.
  `effects_per_group` Hessian-vector products of the joint density give all blocks at
  once, because groups do not interact once the population values are fixed.
- **Marginal fit.** Laplace's formula turns those modes and blocks into the marginal
  density of the population values. A damped Newton method maximizes it with its exact
  gradient and Hessian. The fit is repeated from `num_fits` starting values;
  `num_converged` and `num_agreeing` report how many converged and how many of those
  ended at the chosen point.
- **Draws.** Each chain's population values start as a draw from the normal
  distribution at the fit and are then updated on the marginal itself by slice
  sampling, one whitened coordinate at a time (`num_sweeps=1`). With few groups the
  marginal of a scale parameter is far from normal, and this update removes most of
  that error. The chain's effects are drawn from the normal distribution at their
  conditional mode. `inverse_mass` estimates the variance of the starts per
  coordinate: the fitted covariance carried through the conditional modes to first
  order, plus the conditional variance at the fit.

The joint density is never maximized over all coordinates together: the effects only
move to their conditional mode for given population values. A group scale that
collapses to zero attracts an optimizer of the joint density; here it plays no role.
The starts also do not depend on whether the effects are written centered or scaled.

When the effects are conditionally normal, the marginal density and the conditional
draws are exact. Otherwise both are Laplace approximations.

The one-factor fit assumes independent groups given the population parameters.
- **Many population-level parameters.** The fit takes one directional derivative of
  the marginal per parameter.
- **Parameters with their own scale.** Only a parameter whose prior does not depend on
  other unknowns is population-level. One that has its own scale parameter is an
  effect; treated as population-level it sends the fit into a collapsed scale.

The slice update takes a few dozen evaluations of the marginal per chain. With
thousands of effects it is the largest part of the cost, and with that many groups it
changes little; `num_sweeps=0` skips it.

An error is raised if none of the fits converges. That includes a population-level
parameter that neither the data nor its prior determines, and effects without a
conditional mode. Fits that do not converge are left out. Converged fits that end at
different points produce a warning, and the one with the highest marginal density is
used. Enable 64-bit floats in JAX before calling.

### Two crossed or nested grouping factors

A response may involve a level from each of two factors, for example a person and an
item, or a cask and its enclosing batch. Pass two ordered `GroupFactor` objects.
Each contains a table of unconstrained coordinate indices, one row per level, and
an integer vector giving the zero-based level of each observation. Allocate separate
level IDs for subgroups belonging to different parents. Population coordinates and
both tables must cover the flat position exactly once.

For example, these observations have a shared intercept, person and item effects
with known normal priors, and unit observation noise:

```python
from batched_nuts import GroupedDensity, GroupFactor, hierarchical_start

person = np.array([0, 1, 0, 2, 2])
item = np.array([0, 0, 1, 1, 1])
response = jnp.array([1.0, -0.1, 0.8, 0.2, 0.3])
groups = [
    GroupFactor(np.array([[1], [2], [3]]), person),
    GroupFactor(np.array([[4], [5]]), item),
]
model = GroupedDensity(
    population=lambda h, d: -0.5 * h[0] ** 2,
    normal=lambda h, d: ((0.0, 1.0), (0.0, 0.5)),
    observation=lambda h, a, b, i, d: -0.5 * (d[i] - h[0] - a[0] - b[0]) ** 2,
    quadratic=True,
)
start = hierarchical_start(model, [0], groups, 128, jax.random.key(0), data=response)
```

The population callback supplies the log density of the population parameters in
their unconstrained coordinates, including any transformation Jacobian. The normal
callback returns a location and positive standard deviation for each coordinate
of each factor, broadcastable to its table. Given the population values, these
priors are independent. The observation callback returns one row’s log likelihood,
including its scale-dependent normalization. Use the matching full joint log
density when running NUTS.

The first factor is integrated independently by level. Set `quadratic=True` only
when every observation log likelihood is jointly quadratic in the two effects at
fixed population values. Otherwise the first effect must be scalar and its
observation log likelihood concave in that effect. The algorithm integrates it
by numerical quadrature, fits a conditional Gaussian to the second factor, and
adjusts its mean using directional third derivatives. Nested effects use separate
curvature blocks. Crossed effects use sparse observation-to-level coupling, with
a bounded determinant approximation when more than eight singular directions
contribute. The returned diagonal inverse mass is estimated from the generated
ensemble, using at least 16 draws. This path uses CPU-side sparse algebra and one
population fit starts at zero by default. A single `GroupFactor` in a list accepts
the ordinary joint log density and uses the original one-factor algorithm.

## What you provide, and what the package leaves out

- **Step size and mass matrix.** Every chain has a fixed step size and a fixed diagonal
  or structured inverse mass matrix during sampling. Supply them in `Jobs` or use the jobs returned
  by `warmup`. The benchmark uses `blackjax.window_adaptation` for its common start.
- **Positions are vectors.** Flatten a structured position yourself, for example with
  `jax.flatten_util.ravel_pytree`.
- **Starting positions.** Supply them in `Jobs`. For supported hierarchical models, `hierarchical_start` computes them together with an inverse mass.
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

`tests/test_hierarchical.py` checks `hierarchical_start` on three simulated models.

- **A normal posterior is drawn exactly.** The fitted values are within 1e-4 standard
  errors of the exact ones, their covariance and the inverse mass equal the exact ones
  to a relative 1e-6, and 4,000 starts whitened by the exact posterior have zero mean
  and unit covariance within sampling error.
- **The parametrization of the effects does not matter.** A model with one effect per
  group and a group scale gives the same starts whether the effects are written
  centered or scaled.
- **A marginal that is not normal is followed.** With 8 groups the logarithm of the
  group scale has a skewed marginal. Against that marginal computed exactly on a
  grid, 4,000 starts have its mean to 0.1 standard deviations and its variance to
  15%. Without the slice update the variance is below 0.7 of the exact one.
- **Effects that enter nonlinearly.** On an exponential decay with two effects per
  group, the fit is a stationary point of an independent Laplace objective, built with
  a generic optimizer and a dense Hessian, and has its curvature to 2%.
- **Failures are reported.** A marginal with two modes gives a warning that fits ended
  at different points. A population-level parameter that nothing determines, and
  effects whose stationary point at zero is not a mode, raise an error.

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

### Warmup events

Pass a host callback to observe a chain before its first draw. This example
records when each chain finishes warmup while counting the draws received.

```python
freeze_rounds = {}

def record_warmup(events):
    for event in events:
        if event.frozen:
            freeze_rounds[event.job] = event.round

draw_count = 0
for batch in sample_batches(logdensity, jobs, schedule=AdaptiveWindows(),
                            on_warmup=record_warmup, max_rounds=50_000):
    draw_count += len(batch.records.job)
```

The callback receives a tuple of up to 64 `WarmupEvent` records from one
leapfrog round, in job order. It runs synchronously as the iterator advances,
before the draw batch for that round is yielded. Callback exceptions propagate
to the caller. Draw order, batch boundaries and the first draw round are the
same with diagnostics enabled or disabled. A run with fitted jobs skips warmup.

- `job` and `window` are zero-based. `round` counts leapfrogs from the start
  of the run. One record covers all events for that job in that round.
  `divergent` marks a divergent warmup transition, including one before the
  first window closes. `window_closed` marks the last transition in the window.
  `frozen` marks completion of final calibration and `failed` marks exhaustion.
- `num_evaluations`, `num_transitions` and `num_divergences` count work in
  this window through the event. At an adaptive metric-window close,
  `stable` gives the stopping decision once a comparison is possible.
  For a diagonal comparison, `stability_range` gives the actual range compared
  with `log(schedule.tolerance)`. An attempted nonfinite comparison preserves
  its nonfinite range and gives `stable=False`. The first window has no previous
  shape, and fixed windows and final calibration have no stability test. Fields
  for inapplicable comparisons are `None`.
- At an adaptive diagonal window close, `evidence_source` names the trajectory
  evidence used by the stopping monitor. Pass `include_window_evidence=True`
  to receive its `WindowEvidence` before reset, including the first window.
  It holds the weight and the coordinatewise means and variances of positions
  and gradients. Other events have `evidence=None`. Each event records
  `evidence_requested`; when archiving a run, save the options and last
  processed round as well, including for runs that produced no events.

The default path allocates no event buffer and transfers no diagnostics.
Summary callbacks carry scalars per lane. Requesting moments additionally
uses per-lane storage proportional to the parameter dimension and transfers
only closing window rows, in bounded groups. The sampler retains no event history.

### Branch from a warmed chain

Pass `branching` to widen a running stream. The jobs supplied at startup are
the original chains. A child starts a new trajectory on its parent's dataset,
with the parent's frozen step size and inverse mass and a fresh random key.
The parent continues its current trajectory. Each subsequent gradient round
advances the wider batch together.

This example starts two children after at least two original chains on the same
dataset have frozen and supplied a post-warmup draw. A caller tracking mixing,
time remaining or measured gradient latency can return an empty sequence until
its own criteria are met.

```python
from batched_nuts import AdaptiveWindows, Branch, Branching, sample_batches

def choose_children(batch, parents, room):
    available = [p for p in parents if p.draw_round is not None]
    by_dataset = {}
    for p in available:
        group = by_dataset.setdefault(p.dataset, [])
        group.append(p)
        if len(group) == 2:
            return [Branch(parent.job, parent.draw_round) for parent in group[:room]]
    return []

branch_key = jax.random.key(37)
policy = Branching(key=branch_key, ready_target=2,
                   max_chains=len(jobs.position) + 2, choose=choose_children)
for batch in sample_batches(logdensity, jobs, data=data, schedule=AdaptiveWindows(),
                            branching=policy, rounds_per_batch=50, max_rounds=50_000):
    for origin in batch.branches:
        print(origin.job, origin.parent, origin.source_round, origin.admission_round)
```

Supply a separate `branch_key` from the random keys used for the original jobs.
Child IDs follow the original job IDs in admission order. Child `j` uses
`jax.random.fold_in(branch_key, j)`. Each `BranchOrigin` records the dataset,
whether the source was a warmup endpoint, and the number of discarded child
transitions. The first retained child transition has draw number zero and its
actual completion round on the original stream clock.

Set `warmup_history=h` to keep the last `h` completed warmup transition
positions per original chain. The bounded device buffer includes the final
calibration endpoint. The sampler copies these points to the host when the
parent freezes. The callback can select one with
`Branch(p.job, p.warmup_rounds[-1], settle=n)`. Choose the settling period
explicitly for a warmup source. The child runs its first `n` completed
transitions with the inherited tuning before contributing draws. The source
position itself contributes no extra draw. With the default history size of
zero, only the most recent retained draw per eligible parent is available.

The callback receives a draw batch, eligible parents and the remaining capacity
at freezes, buffer flushes and round checkpoints. It may decline repeatedly or
request several children from the same source. The readiness threshold counts
original chains separately for each dataset. Children share their parent's
history, so they do not provide independent starting points for diagnosing
mixing. Failed chains cannot become parents. Fitted jobs supplied without a
warmup schedule count as ready at startup. At the global round limit the
sampler returns the last records and stops admitting children.

Early gradient calls use the initial width. Each distinct wider shape can incur
a compilation cost. An explicit draw buffer must hold `max_chains` simultaneous
records. Without an explicit size, the buffer grows if the actual width
outgrows it. `lower_sample_batches(..., branching=policy)` inspects the actual
startup and first round at the initial width, including optional history storage.
It never calls the admission policy. Omitting `branching` uses the usual
fixed-width `DrawBatch` stream.

### Inspect the stream before sampling

`lower_sample_batches` accepts the same density, jobs and keyword arguments as
`sample_batches`. It traces startup and the first round loop on abstract sampler
state. You can inspect the compiler input or compile either stage without running
a chain or allocating its trajectory and draw buffers.

```python
from batched_nuts import AdaptiveWindows, lower_sample_batches

plans = lower_sample_batches(logdensity, jobs, schedule=AdaptiveWindows(),
                             buffer_size=1024, max_rounds=50_000)
for stage, program in plans._asdict().items():
    memory = program.compile().memory_analysis()
    if memory is not None:
        print(stage, "planned peak bytes", memory.peak_memory_in_bytes)
```

`plans.startup` initializes the chains. With a warmup schedule, `plans.round`
advances warming and sampling chains together. With fitted jobs it advances
sampling alone. Each is a JAX `Lowered` object with `as_text()` for inspection.
With scheduled warmup, passing `on_warmup` selects the event-producing program.
Inspection never calls the callback. At `max_rounds=0` the stream needs neither
program and the function returns `None`. Memory analysis depends on the compiler
and backend. Its byte counts
describe the compiler's plan. Measuring physical device residency requires
observing a running process. When JAX automatically partitions a dataset across a
device mesh, inspection compiles startup to obtain the layouts for the round
loop. The sampler still runs only when you advance `sample_batches`.

## License

MIT
