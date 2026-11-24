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
    _begin_doubling,
    _begin_next,
    _begin_transition,
    _empty_lanes,
    _prepare_queue,
    _where,
)
from batched_nuts.policy import AdaptiveWindows, DiagonalAdaptation, FixedWindows, _choice


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


class Evidence(NamedTuple):
    moments: Moments
    weight: Any


def _empty_evidence(point):
    return Evidence(point, jnp.int32(0))


def _observe(state, observation, variance):
    """Add unit observations, using their conditional variance when supplied."""
    weight = state.weight + observation.weight
    p = observation.weight.astype(state.moments.mean.dtype) / jnp.maximum(weight, 1)
    old, new = state.moments, observation.moments
    if variance == "population":
        moments = _mix(old, new, p)
    else:
        # The pre-update-mean scatter used in nutpie adds (x - old_mean)^2.
        delta = new.mean - old.mean
        mixed = Moments(old.mean + p * delta, (1 - p) * old.var + p * (new.var + delta**2))
        moments = _where(p == 0, old, _where(p == 1, new, mixed))
    return Evidence(moments, weight)


class Indices(NamedTuple):
    left: Any
    right: Any
    subtree: Any
    selected: Any


class TreeEvidence(NamedTuple):
    subtree: Moments
    trajectory: Moments
    indices: Indices
    symmetric_sum: Any


class Adaptation(NamedTuple):
    step: Any
    metric: Any
    dual: DualAveraging
    tree: TreeEvidence
    active: Evidence
    background: Evidence
    switch_work: Any
    update_work: Any
    window: Evidence
    last_shape: Any
    phase: Any
    failed: Any
    evaluations: Any
    transitions: Any
    divergences: Any


def _initial(position, grad, step, metric, n_windows, include_initial=False):
    point = _point(position, grad)
    zero = jnp.int32(0)
    tree = TreeEvidence(point, point, Indices(zero, zero, zero, zero), 0 * step)
    start = Evidence(point, jnp.int32(include_initial))
    counts = jnp.zeros(n_windows, jnp.int32)
    return Adaptation(
        step,
        metric,
        _start_dual(step),
        tree,
        start,
        start,
        zero,
        zero,
        _empty_evidence(point),
        jnp.full_like(metric, jnp.nan),
        zero,
        False,
        counts,
        counts,
        counts,
    )


def _log_diagonal(var, estimator):
    position, score = jnp.split(var, 2)
    if estimator == "position":
        return jnp.log(position)
    return 0.5 * (jnp.log(position) - jnp.log(score))


def _inverse_mass(var, old, count, estimator="position_score", shrinkage=3.0):
    log_fit = _log_diagonal(var, estimator)
    log_old = jnp.log(old)
    weight = count / jnp.maximum(count + shrinkage, 1)
    value = jnp.exp(log_old + weight * (log_fit - log_old))
    return jnp.where(jnp.isfinite(value) & (value > 0), value, old)


def _track_tree(
    previous,
    sources,
    direction,
    leaf,
    chosen,
    delta,
    probabilities,
    take_leaf,
    take_tree,
    ended,
    divergent,
    acceptance,
):
    """Share leaf arithmetic between the fitting estimate and its independent monitor."""
    observations = {}
    subtree, trajectory, indices, symmetric_sum = previous
    for source in sources:
        if source == "eligible_leaves":
            valid = (delta > -jnp.log(100.0)) & jnp.all(jnp.isfinite(leaf.mean))
            observations[source] = Evidence(leaf, valid.astype(jnp.int32))
        elif source == "nuts":
            p_leaf, p_tree = probabilities
            subtree = _mix(subtree, leaf, p_leaf)
            trajectory = _mix(trajectory, subtree, p_tree)
            observations[source] = Evidence(trajectory, ended.astype(jnp.int32))
        elif source == "eligible_selected":
            left, right, sub, selected = indices
            left -= (direction < 0).astype(jnp.int32)
            right += (direction > 0).astype(jnp.int32)
            sub = jnp.where(take_leaf, jnp.where(direction < 0, left, right), sub)
            selected = jnp.where(take_tree, sub, selected)
            valid = jnp.where(divergent, jnp.abs(selected) > 4, selected != 0)
            observations[source] = Evidence(chosen, (ended & valid).astype(jnp.int32))
            indices = _where(
                ended,
                jax.tree.map(jnp.zeros_like, indices),
                Indices(left, right, sub, selected),
            )
        else:
            observations[source] = Evidence(chosen, ended.astype(jnp.int32))
    if acceptance != "mean":
        symmetric_sum += 2 * jax.nn.sigmoid(-jnp.abs(delta))
    tree = TreeEvidence(
        subtree,
        _where(ended, chosen, trajectory),
        indices,
        jnp.where(ended, 0, symmetric_sum),
    )
    return tree, observations, symmetric_sum


def _clock(clock, steps, transitions):
    return steps if clock == "gradients" else transitions


def _warm_step(
    logdensity,
    max_depth,
    policy,
    schedule,
    acceptance,
    restart_dual,
    budgets,
    target,
    chain,
    lane,
    stats,
):
    chain = chain._replace(step_size=stats.step, inverse_mass=stats.metric)
    lane, rec, ended, sub_done, leaf, prob, delta, take_leaf, take_tree = _advance(
        logdensity, max_depth, chain, lane, warmup_info=True
    )
    adaptive = isinstance(schedule, AdaptiveWindows)
    monitor = (schedule.evidence or policy.evidence) if adaptive else policy.evidence
    sources = tuple(dict.fromkeys((policy.evidence, monitor)))
    selected = _point(lane.proposal.position, lane.proposal.grad)
    tree, observations, symmetric_sum = _track_tree(
        stats.tree,
        sources,
        lane.direction,
        _point(leaf.position, leaf.grad),
        selected,
        delta,
        prob,
        take_leaf,
        take_tree,
        ended,
        rec.is_divergent,
        acceptance,
    )
    observation = observations[policy.evidence]
    empty = _empty_evidence(selected)

    phase = stats.phase
    last = budgets.size - 1
    final = phase == last
    evaluations = stats.evaluations[phase] + rec.num_integration_steps
    transitions = stats.transitions[phase] + 1
    cost = _clock(schedule.clock, evaluations, transitions)
    final_cost = _clock(schedule.final_clock or schedule.clock, evaluations, transitions)
    boundary = ended & (jnp.where(final, final_cost, cost) >= budgets[phase])
    window, shape = stats.window, stats.last_shape
    stable = jnp.bool_(False)
    if adaptive:
        # Fit the diagnostic separately on disjoint windows, even if the installed
        # diagonal keeps all its history or uses a different trajectory source.
        window = _observe(window, observations[monitor], policy.variance)
        candidate = _log_diagonal(window.moments.var, policy.estimator)
        change = 0.5 * (candidate - shape)
        stable = jnp.all(jnp.isfinite(change)) & (
            jnp.max(change) - jnp.min(change) < jnp.log(schedule.tolerance)
        )
        shape = jnp.where(boundary & ~final, candidate, shape)
        window = _where(boundary, empty, window)

    active = _observe(stats.active, observation, policy.variance)
    background = stats.background
    switch_work = stats.switch_work + _clock(
        policy.switch_clock or schedule.clock, rec.num_integration_steps, 1
    )
    switch = (
        boundary
        if policy.switch_interval == "window"
        else ended & (switch_work >= policy.switch_interval)
    )
    if policy.memory == "overlapping":
        background = _observe(background, observation, policy.variance)
        active = _where(switch & (background.weight > 0), background, active)
        background = _where(switch, empty, background)

    update_work = stats.update_work + _clock(
        policy.update_clock, rec.num_integration_steps, 1
    )
    install = boundary
    if policy.update_interval != "window":
        install |= ended & (update_work >= policy.update_interval)
    metric = stats.metric
    if policy.estimator != "fixed":
        metric = jnp.where(
            install & ~final & (active.weight > 0),
            _inverse_mass(
                active.moments.var,
                metric,
                active.weight,
                policy.estimator,
                policy.shrinkage,
            ),
            metric,
        )
    if policy.memory == "window":
        active = _where(switch, empty, active)

    alpha = rec.acceptance_rate
    if acceptance != "mean":
        use_symmetric = final if acceptance == "late_symmetric" else True
        alpha = jnp.where(use_symmetric, symmetric_sum / rec.num_integration_steps, alpha)
    dual, step = _dual(stats.dual, alpha, target)
    reset = boundary & ~final & restart_dual
    smoothed = jnp.exp(dual.log_average)
    step = jnp.where(reset | (boundary & final), smoothed, step)
    dual = _where(reset, _start_dual(smoothed), dual)
    failed = boundary & (phase == last - 1) & ~stable if adaptive else jnp.bool_(False)
    new_phase = jnp.where(boundary & ~final & stable, last, phase + boundary)
    next_stats = Adaptation(
        jnp.where(ended, step, stats.step),
        metric,
        _where(ended, dual, stats.dual),
        tree,
        _where(final, stats.active, active),
        _where(final, stats.background, background),
        jnp.where(ended & ~final, jnp.where(switch, 0, switch_work), stats.switch_work),
        jnp.where(ended & ~final, jnp.where(install, 0, update_work), stats.update_work),
        _where(final, stats.window, window),
        shape,
        new_phase,
        failed,
        stats.evaluations.at[phase].add(jnp.where(ended, rec.num_integration_steps, 0)),
        stats.transitions.at[phase].add(ended.astype(jnp.int32)),
        stats.divergences.at[phase].add((ended & rec.is_divergent).astype(jnp.int32)),
    )
    updated = chain._replace(step_size=next_stats.step, inverse_mass=metric)
    lane = _begin_next(updated, lane, ended, sub_done)
    return lane._replace(active=lane.active & ~(boundary & final) & ~failed), next_stats


@lru_cache(maxsize=16)
def _compiled_warmup(logdensity, max_depth, policy, schedule, acceptance, restart_dual):
    n_windows = len(schedule.lengths())

    def refill(lanes, chains, stats, take, job, rows, data):
        keys = jax.vmap(jax.random.split)(jax.random.wrap_key_data(rows.key))
        rows = rows._replace(key=jax.random.key_data(keys[:, 0]))
        lanes, chains = _assign(logdensity, lanes, chains, take, job, rows, data)
        if policy.initial == "gradient":
            grad = jnp.abs(lanes.proposal.grad)
            limits = jnp.finfo(grad.dtype)
            low, high = max(1e-20, limits.tiny), min(1e20, limits.max)
            metric = jnp.where(grad > 0, jnp.clip(1 / grad, low, high), chains.inverse_mass)
            chains = chains._replace(
                inverse_mass=jnp.where(take[:, None], metric, chains.inverse_mass)
            )
            started = jax.vmap(
                lambda chain, lane: _begin_doubling(_begin_transition(chain, lane))
            )(chains, lanes)
            lanes = jax.vmap(_where)(take, started, lanes)
        initial = jax.vmap(
            partial(_initial, n_windows=n_windows, include_initial=policy.include_initial)
        )(
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
        advance = jax.vmap(
            partial(
                _warm_step,
                logdensity,
                max_depth,
                policy,
                schedule,
                acceptance,
                restart_dual,
                budgets,
                target,
            )
        )

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
    window_evaluations=None,
    diagonal=None,
    schedule=None,
    acceptance="mean",
    restart_dual=True,
    target_acceptance=0.8,
    data=None,
    num_lanes=None,
    max_depth=10,
) -> WarmupResult:
    """Learn one diagonal inverse mass and step size for each chain.

    Choose the diagonal estimator with `diagonal` and the stopping rule with a
    `FixedWindows` or `AdaptiveWindows` schedule. As shorthand, `window_evaluations`
    gives fixed per-chain gradient budgets, including the final step-size-only window.
    The default budgets are (1000, 2000, 1000). A chain always completes its trajectory.

    `acceptance` selects mean tree acceptance, symmetric acceptance, or symmetric
    acceptance only in the final window. `restart_dual` restarts dual averaging at
    metric-window boundaries. Sampling receives frozen parameters and independent keys.
    Per-window counts exclude initialization. Adaptive schedules leave unused metric
    windows at zero and store final calibration counts in the last column.
    """
    if diagonal is None:
        diagonal = DiagonalAdaptation()
    if not isinstance(diagonal, DiagonalAdaptation):
        raise TypeError("diagonal must be a DiagonalAdaptation.")
    if schedule is not None and window_evaluations is not None:
        raise ValueError("Specify either schedule or window_evaluations.")
    if schedule is None:
        if window_evaluations is None:
            schedule = FixedWindows()
        else:
            lengths = tuple(map(index, window_evaluations))
            if len(lengths) < 2:
                raise ValueError("Supply at least two positive window budgets.")
            schedule = FixedWindows(lengths[:-1], lengths[-1])
    if not isinstance(schedule, (FixedWindows, AdaptiveWindows)):
        raise TypeError("schedule must be FixedWindows or AdaptiveWindows.")
    if isinstance(schedule, AdaptiveWindows) and diagonal.estimator == "fixed":
        raise ValueError("A metric-stability endpoint requires a learned diagonal.")
    _choice("acceptance", acceptance, ("mean", "symmetric", "late_symmetric"))
    budgets = schedule.lengths()
    max_depth = index(max_depth)
    if not 1 <= max_depth <= 30:
        raise ValueError("max_depth must be between 1 and 30.")
    longest = 2**max_depth - 1
    clocks = (schedule.clock,) * (len(budgets) - 1) + (
        schedule.final_clock or schedule.clock,
    )
    max_evaluations = 0
    max_transitions = 0
    for budget, clock in zip(budgets, clocks, strict=True):
        upper = budget + longest - 1 if clock == "gradients" else budget * longest
        max_evaluations += upper
        max_transitions += budget
        if upper > np.iinfo(np.int32).max:
            raise ValueError("Window budgets exceed the supported int32 count range.")
    # The PRNG draw index and cumulative evidence counters also live on the device.
    if max_transitions >= np.iinfo(np.int32).max or (
        diagonal.evidence == "eligible_leaves"
        and diagonal.memory != "window"
        and max_evaluations >= np.iinfo(np.int32).max
    ):
        raise ValueError("The total warmup budgets exceed the int32 count range.")
    if not isinstance(restart_dual, (bool, np.bool_)):
        raise TypeError("restart_dual must be a boolean.")
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
    refill, run = _compiled_warmup(
        logdensity, max_depth, diagonal, schedule, acceptance, restart_dual
    )
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
        if isinstance(schedule, AdaptiveWindows):
            failed = job_ids[active & np.asarray(stats.failed)]
            if len(failed):
                raise RuntimeError(
                    f"Metric stability not reached within max_windows for jobs {failed}."
                )
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
