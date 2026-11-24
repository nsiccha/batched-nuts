"""Per-chain diagonal metric and step-size adaptation, batched by leapfrog step."""

from functools import lru_cache, partial
from operator import index
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from batched_nuts.nuts import (
    Jobs,
    _advance,
    _assign,
    _assignment,
    _begin_next,
    _empty_lanes,
    _prepare_queue,
    _where,
)


class WarmupResult(NamedTuple):
    """Frozen sampling jobs and counts with shape (number of chains, number of windows)."""

    jobs: Jobs
    num_evaluations: np.ndarray
    num_transitions: np.ndarray
    num_divergences: np.ndarray


class Moments(NamedTuple):
    mean: Any
    var: Any


def _point(position, grad):
    mean = jnp.concatenate((position, grad))
    return Moments(mean, jnp.zeros_like(mean))


def _mix(old, new, p):
    """Centered moments of a mixture, including exact zero and unit probabilities."""
    delta = new.mean - old.mean
    mixed = Moments(
        old.mean + p * delta,
        (1 - p) * old.var + p * new.var + p * (1 - p) * delta**2,
    )
    return _where(p == 0, old, _where(p == 1, new, mixed))


class DualAveraging(NamedTuple):
    center: Any
    iteration: Any
    error_sum: Any
    log_average: Any


def _start_dual(step):
    log_step = jnp.log(step)
    return DualAveraging(log_step + jnp.log(10.0), jnp.int32(0), 0 * step, log_step)


def _dual(state, acceptance, target):
    t = state.iteration + 1
    error_sum = state.error_sum + target - acceptance
    log_step = (
        state.center - jnp.sqrt(t.astype(error_sum.dtype)) / (0.05 * (t + 10)) * error_sum
    )
    # Keep the step representable even after a long sequence of rejections.
    limits = jnp.finfo(state.log_average.dtype)
    log_step = jnp.clip(log_step, jnp.log(limits.tiny) + 1, jnp.log(limits.max) - 1)
    log_average = state.log_average + t.astype(error_sum.dtype) ** -0.75 * (
        log_step - state.log_average
    )
    return DualAveraging(state.center, t, error_sum, log_average), jnp.exp(log_step)


class Adaptation(NamedTuple):
    step: Any
    metric: Any
    dual: DualAveraging
    subtree: Moments
    trajectory: Moments
    window: Moments
    phase: Any
    evaluations: Any
    transitions: Any
    divergences: Any


def _initial(position, grad, step, metric, n_windows):
    point = _point(position, grad)
    counts = jnp.zeros(n_windows, jnp.int32)
    return Adaptation(
        step,
        metric,
        _start_dual(step),
        point,
        point,
        point,
        jnp.int32(0),
        counts,
        counts,
        counts,
    )


def _inverse_mass(var, old, count):
    position_var, score_var = jnp.split(var, 2)
    log_ratio = 0.5 * (jnp.log(position_var) - jnp.log(score_var))
    # Shrink a short window toward the previous diagonal; preserve an unidentified scale.
    log_old = jnp.log(old)
    value = jnp.exp(log_old + count / (count + 3.0) * (log_ratio - log_old))
    valid = jnp.isfinite(value) & (value > 0)
    return jnp.where(valid, value, old)


def _warm_step(logdensity, max_depth, budgets, target, chain, lane, stats):
    chain = chain._replace(step_size=stats.step, inverse_mass=stats.metric)
    lane, record, ended, sub_done, leaf, (p_leaf, p_tree) = _advance(
        logdensity, max_depth, chain, lane
    )
    subtree = _mix(stats.subtree, _point(leaf.position, leaf.grad), p_leaf)
    trajectory = _mix(stats.trajectory, subtree, p_tree)
    phase = stats.phase
    count = stats.transitions[phase] + 1
    # Every trajectory has unit weight, regardless of how many leaves it visited.
    window = _mix(stats.window, trajectory, 1.0 / count.astype(record.position.dtype))
    cost = stats.evaluations[phase] + record.num_integration_steps
    boundary = ended & (cost >= budgets[phase])
    final = phase == budgets.size - 1

    candidate = _inverse_mass(window.var, stats.metric, count)
    # Install this window’s estimate before the next momentum draw.
    refit = boundary & ~final
    metric = jnp.where(refit, candidate, stats.metric)
    dual, step = _dual(stats.dual, record.acceptance_rate, target)
    smoothed = jnp.exp(dual.log_average)
    step = jnp.where(refit | (boundary & final), smoothed, step)
    dual = _where(refit, _start_dual(smoothed), dual)
    start = _point(lane.proposal.position, lane.proposal.grad)
    next_stats = Adaptation(
        jnp.where(ended, step, stats.step),
        metric,
        _where(ended, dual, stats.dual),
        subtree,
        _where(ended, start, trajectory),
        _where(boundary, start, _where(ended, window, stats.window)),
        phase + boundary.astype(jnp.int32),
        stats.evaluations.at[phase].add(jnp.where(ended, record.num_integration_steps, 0)),
        stats.transitions.at[phase].add(ended.astype(jnp.int32)),
        stats.divergences.at[phase].add((ended & record.is_divergent).astype(jnp.int32)),
    )
    updated = chain._replace(step_size=next_stats.step, inverse_mass=metric)
    lane = _begin_next(updated, lane, ended, sub_done)
    return lane._replace(active=lane.active & ~(boundary & final)), next_stats


@lru_cache(maxsize=16)
def _compiled_warmup(logdensity, max_depth, n_windows):
    def refill(lanes, chains, stats, take, job, rows, data):
        keys = jax.vmap(jax.random.split)(jax.random.wrap_key_data(rows.key))
        rows = rows._replace(key=jax.random.key_data(keys[:, 0]))
        lanes, chains = _assign(logdensity, lanes, chains, take, job, rows, data)
        initial = jax.vmap(partial(_initial, n_windows=n_windows))(
            lanes.proposal.position,
            lanes.proposal.grad,
            chains.step_size,
            chains.inverse_mass,
        )
        valid = jnp.isfinite(lanes.proposal.logdensity) & jnp.all(
            jnp.isfinite(lanes.proposal.grad), axis=1
        )
        return (
            lanes,
            chains,
            jax.vmap(_where)(take, initial, stats),
            jax.random.key_data(keys[:, 1]),
            valid,
        )

    def run(lanes, stats, chains, budgets, target, waiting):
        active_at_start = lanes.active
        advance = jax.vmap(partial(_warm_step, logdensity, max_depth, budgets, target))

        def iteration(state):
            next_state = advance(chains, *state)
            return jax.vmap(_where)(state[0].active, next_state, state)

        def busy(state):
            active = state[0].active
            return jnp.any(active) & (~waiting | jnp.all(active == active_at_start))

        return jax.lax.while_loop(busy, iteration, (lanes, stats))

    return jax.jit(refill), jax.jit(run)


def warmup(
    logdensity,
    jobs: Jobs,
    *,
    window_evaluations=(1000, 2000, 1000),
    target_acceptance=0.8,
    data=None,
    num_lanes=None,
    max_depth=10,
) -> WarmupResult:
    """Learn one diagonal inverse mass and step size for each chain.

    Each positive entry in `window_evaluations` is a per-chain leapfrog budget.
    The chain finishes the trajectory crossing that budget. Every window except the
    last fits the metric. The final window only tunes the step size. Pass at least two
    windows. `jobs.step_size` and `jobs.inverse_mass` supply the starting guesses.

    Return frozen jobs with independent sampling keys, ready for `sample`, together
    with per-chain, per-window counts. Evaluations count useful leapfrog steps and
    exclude initialization. Data and lane assignment follow the same rules as `sample`.
    """
    budgets = tuple(map(index, window_evaluations))
    max_depth = index(max_depth)
    if not 1 <= max_depth <= 30:
        raise ValueError("max_depth must be between 1 and 30.")
    if len(budgets) < 2 or any(b <= 0 or b > 2**31 - 2**max_depth for b in budgets):
        raise ValueError("Supply at least two positive window budgets within int32 range.")
    if not 0 < target_acceptance < 1:
        raise ValueError("target_acceptance must lie strictly between 0 and 1.")
    logdensity, queue = _prepare_queue(logdensity, jobs, data)
    n_jobs, dim = queue.position.shape
    width = n_jobs if num_lanes is None else min(index(num_lanes), n_jobs)
    if width < 1 or dim < 1:
        raise ValueError("Warmup requires a chain, a coordinate and a lane.")
    for name, shape in (("step_size", (n_jobs,)), ("inverse_mass", (n_jobs, dim))):
        arr = getattr(queue, name)
        if arr.shape != shape or not np.all(np.isfinite(arr) & (arr > 0)):
            raise ValueError(
                f"jobs.{name} must have shape {shape} and finite positive entries."
            )
    if not np.all(np.isfinite(queue.position)):
        raise ValueError("Starting positions must be finite.")
    dtype = queue.position.dtype
    n_windows = len(budgets)
    lanes, chains = _empty_lanes(width, dim, dtype, max_depth, data)
    stats = jax.vmap(partial(_initial, n_windows=n_windows))(
        lanes.proposal.position,
        lanes.proposal.grad,
        jnp.ones(width, dtype),
        jnp.ones((width, dim), dtype),
    )
    keys = np.empty_like(queue.key)
    frozen = Jobs(
        keys,
        np.empty_like(queue.position),
        np.empty_like(queue.step_size),
        np.empty_like(queue.inverse_mass),
        jobs.dataset,
    )
    counts = [np.empty((n_jobs, n_windows), np.int32) for _ in range(3)]
    refill, run = _compiled_warmup(logdensity, max_depth, n_windows)
    next_job = 0
    active = np.zeros(width, bool)
    job_ids = np.zeros(width, np.int32)
    while next_job < n_jobs or active.any():
        assignment = _assignment(active, queue, next_job)
        if assignment is not None:
            take, ids, rows, next_job = assignment
            lanes, chains, stats, sample_keys, valid = refill(
                lanes, chains, stats, take, ids, rows, data
            )
            bad = ids[take & ~np.asarray(valid)]
            if len(bad):
                raise ValueError(
                    f"Nonfinite initial log density or gradient for jobs {bad}."
                )
            active[take] = True
            job_ids[take] = ids[take]
            keys[ids[take]] = np.asarray(sample_keys)[take]
        lanes, stats = run(
            lanes,
            stats,
            chains,
            jnp.asarray(budgets, jnp.int32),
            jnp.asarray(target_acceptance, dtype),
            next_job < n_jobs,
        )
        remaining = np.asarray(lanes.active)
        finished = np.flatnonzero(active & ~remaining)
        ids = job_ids[finished]
        frozen.position[ids] = np.asarray(lanes.proposal.position[finished])
        frozen.step_size[ids] = np.asarray(stats.step[finished])
        frozen.inverse_mass[ids] = np.asarray(stats.metric[finished])
        for output, values in zip(
            counts, (stats.evaluations, stats.transitions, stats.divergences), strict=True
        ):
            output[ids] = np.asarray(values[finished])
        active = remaining.copy()
    return WarmupResult(frozen, *counts)
