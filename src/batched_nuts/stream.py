"""Stream completed draws while every chain continues to advance."""

from collections.abc import Callable, Iterator
from functools import lru_cache, partial
from operator import index
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from batched_nuts.nuts import (
    Jobs,
    Record,
    _broadcast_data,
    _chain_data,
    _compiled,
    _empty,
    _prepare_queue,
    _write_records,
)
from batched_nuts.policy import AdaptiveWindows, DiagonalAdaptation
from batched_nuts.warmup import (
    _compiled_warmup,
    _initial,
    _prepare_warmup,
    _StepEvents,
    _warm_step,
)


class DrawBatch(NamedTuple):
    """Host records in completion order and cumulative leapfrog rounds since start."""

    records: Record
    num_rounds: int
    completion_rounds: np.ndarray
    failed_jobs: np.ndarray


class LoweredStream(NamedTuple):
    """JAX programs for startup and the first round loop of a draw stream."""

    startup: jax.stages.Lowered
    round: jax.stages.Lowered


class WindowEvidence(NamedTuple):
    """Moments and weight from the stopping monitor's just-closed diagonal window."""

    weight: int
    position_mean: np.ndarray
    position_variance: np.ndarray
    gradient_mean: np.ndarray
    gradient_variance: np.ndarray


class WarmupEvent(NamedTuple):
    """One chain at one event round. Counts include this transition in this window."""

    job: int
    round: int
    window: int
    window_closed: bool
    divergent: bool
    frozen: bool
    failed: bool
    num_evaluations: int
    num_transitions: int
    num_divergences: int
    stable: bool | None
    stability_range: float | None
    evidence_source: str | None
    evidence_requested: bool
    evidence: WindowEvidence | None


@lru_cache(maxsize=16)
def _mixed_loop(
    target,
    max_depth,
    policy,
    schedule,
    acceptance,
    restart_dual,
    observe=False,
    window_evidence=False,
):
    def run(state, chains, stats, keys, budgets, target_accept, limit, first):
        state = state._replace(num_iterations=jnp.int32(0))
        if not observe:
            state = state._replace(num_records=jnp.int32(0))
        step = partial(
            _warm_step,
            target,
            max_depth,
            policy,
            schedule,
            acceptance,
            restart_dual,
            budgets,
            target_accept,
            stream=True,
            observe=observe,
            window_evidence=window_evidence,
        )
        width = state.lanes.active.size
        capacity = state.records.job.size
        failed_before = stats.failed

        def go_on(carry):
            state, _, stats, _ = carry
            return (
                (state.num_iterations < limit)
                & (state.num_records + width <= capacity)
                & (~first | (state.num_records == 0))
                & jnp.any(state.lanes.active & (stats.phase < budgets.size))
                & ~jnp.any(stats.failed & ~failed_before)
            )

        def advance(carry):
            state, chains, stats, _ = carry
            result = jax.vmap(
                lambda chain, lane, adapt, key: step(chain, lane, adapt, sampling_key=key),
                axis_name="lanes",
            )(chains, state.lanes, stats, keys)
            chains, lanes, stats, records, done = result[:5]
            events = result[5] if observe else None
            state = state._replace(
                lanes=lanes,
                records=_write_records(state, records, done),
                num_records=state.num_records + jnp.sum(done, dtype=jnp.int32),
                num_iterations=state.num_iterations + 1,
            )
            return state, chains, stats, events

        initial = None
        if observe:
            diagonal_monitor = isinstance(schedule, AdaptiveWindows) and isinstance(
                policy, DiagonalAdaptation
            )
            clear = jnp.zeros_like(stats.failed)
            zero = jnp.zeros_like(stats.phase)
            initial = _StepEvents(
                clear,
                clear,
                zero,
                zero,
                zero,
                zero,
                clear,
                jnp.zeros_like(stats.step) if diagonal_monitor else None,
                stats.fit.window if diagonal_monitor and window_evidence else None,
            )

        def condition(carry):
            running = go_on(carry)
            if observe:
                events = carry[-1]
                running &= ~jnp.any(events.closed | events.divergent)
            return running

        carry = jax.lax.while_loop(condition, advance, (state, chains, stats, initial))
        # An event-only return leaves buffered draws for their original yield point.
        return (*carry, go_on(carry)) if observe else carry[:3]

    return jax.jit(run, donate_argnums=(0, 1, 2))


def _host_record(column, count):
    if count == 0:
        return np.empty((0, *column.shape[1:]), column.dtype)
    if count < column.shape[0] // 4:
        # Early draws can be sparse in thousands of lanes. Limit transfers to
        # power-of-two shapes so the number of distinct slices stays logarithmic.
        column = column[: 1 << (count - 1).bit_length()]
    return np.asarray(column)[:count].copy()


def _host_evidence(evidence, jobs):
    if evidence is None or not len(jobs):
        return {}
    # Gather only closing rows, with power-of-two shapes shared by sparse batches.
    size = 1 << (len(jobs) - 1).bit_length()
    indices = np.pad(jobs, (0, size - len(jobs)), mode="edge")
    gathered = jax.device_get(jax.tree.map(lambda column: column[indices], evidence))
    result = {}
    for i, job in enumerate(jobs):
        position, gradient = np.split(gathered.moments.mean[i], 2)
        pos_var, grad_var = np.split(gathered.moments.var[i], 2)
        result[job] = WindowEvidence(
            int(gathered.weight[i]), position, pos_var, gradient, grad_var
        )
    return result


def _deliver_events(
    callback, events, rounds, failed, last, schedule, policy, window_evidence
):
    info = jax.device_get(events._replace(evidence=None))
    jobs = np.flatnonzero(info.closed | info.divergent)
    adaptive = isinstance(schedule, AdaptiveWindows)
    source = (schedule.evidence or policy.evidence) if info.span is not None else None
    for start in range(0, len(jobs), 64):
        group = jobs[start : start + 64]
        fitting = info.closed[group] & (info.phase[group] < last)
        evidence = _host_evidence(events.evidence, group[fitting])
        batch = []
        for job, fit in zip(group, fitting, strict=True):
            phase = int(info.phase[job])
            compared = adaptive and fit and phase > 0
            batch.append(
                WarmupEvent(
                    job=int(job),
                    round=rounds,
                    window=phase,
                    window_closed=bool(info.closed[job]),
                    divergent=bool(info.divergent[job]),
                    frozen=bool(info.closed[job] and phase == last),
                    failed=bool(failed[job]),
                    num_evaluations=int(info.evaluations[job]),
                    num_transitions=int(info.transitions[job]),
                    num_divergences=int(info.divergences[job]),
                    stable=bool(info.stable[job]) if compared else None,
                    stability_range=float(info.span[job]) if source and compared else None,
                    evidence_source=source if fit else None,
                    evidence_requested=window_evidence,
                    evidence=evidence.get(job),
                )
            )
        callback(tuple(batch))


def _completion_rounds(record, progress):
    # Each chain advances once per round. A stable sort groups its completed
    # trajectories, preserving their order across device-buffer flushes.
    times = np.empty(len(record.job), np.int64)
    if len(times):
        order = np.argsort(record.job, kind="stable")
        ids = record.job[order]
        first = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        last = np.r_[first[1:], len(ids)] - 1
        cumulative = record.num_integration_steps[order].cumsum(dtype=np.int64)
        origin = np.r_[0, cumulative[last[:-1]]]
        times[order] = progress[ids] + cumulative - np.repeat(origin, last - first + 1)
        progress[ids[last]] = times[order[last]]
    return times


def sample_batches(
    logdensity,
    jobs: Jobs,
    *,
    data=None,
    schedule=None,
    diagonal=None,
    structured=None,
    acceptance="mean",
    restart_dual=True,
    target_acceptance=0.8,
    max_depth=10,
    buffer_size=None,
    max_rounds=None,
    rounds_per_batch=None,
    on_warmup: Callable[[tuple[WarmupEvent, ...]], None] | None = None,
    include_window_evidence=False,
) -> Iterator[DrawBatch]:
    """Run one lane per job. Yield bounded host batches without a per-chain draw quota.

    With a warmup schedule, each chain begins sampling when its own adaptation ends.
    Use `structured` or supply a MetricFactor to fit correlated metrics.
    With fitted jobs, omit the schedule. The first completed draw is yielded during
    the round that produced it (possibly with simultaneous draws). Later transfers
    use the record buffer. Set rounds_per_batch to also yield at round checkpoints,
    including empty batches during warmup. Set max_rounds for an exact finite budget,
    or close the iterator to stop. Both limits count leapfrogs, including warmup.
    Each record has its exact trajectory-completion round in completion_rounds.
    The initial batched evaluation of the starting positions precedes round one.
    If a job exhausts its adaptive windows, its ID is yielded once in failed_jobs.
    Other chains continue. The iterator ends if every job exhausts its windows.
    If supplied, on_warmup receives tuples of WarmupEvent on the host at each
    event round, before yielding draws from that round. Set include_window_evidence
    to also receive moments from completed adaptive diagonal metric windows.
    """
    yield from _prepare_stream(
        logdensity,
        jobs,
        data,
        schedule,
        diagonal,
        structured,
        acceptance,
        restart_dual,
        target_acceptance,
        max_depth,
        buffer_size,
        max_rounds,
        rounds_per_batch,
        on_warmup,
        include_window_evidence,
        lower=False,
    )


def lower_sample_batches(
    logdensity,
    jobs: Jobs,
    *,
    data=None,
    schedule=None,
    diagonal=None,
    structured=None,
    acceptance="mean",
    restart_dual=True,
    target_acceptance=0.8,
    max_depth=10,
    buffer_size=None,
    max_rounds=None,
    rounds_per_batch=None,
    on_warmup: Callable[[tuple[WarmupEvent, ...]], None] | None = None,
    include_window_evidence=False,
) -> LoweredStream | None:
    """Lower the startup and first round loop that sample_batches would execute.

    Accepts the same jobs, data and settings as sample_batches. Returns None for
    max_rounds=0. Otherwise each stage can be inspected or compiled separately.
    Lowering traces the density on abstract inputs without advancing a chain or
    allocating the device buffers for its trajectories and draws. With an
    automatically partitioned dataset, lowering compiles startup to determine
    the output layouts that the round loop will receive.
    """
    return _prepare_stream(
        logdensity,
        jobs,
        data,
        schedule,
        diagonal,
        structured,
        acceptance,
        restart_dual,
        target_acceptance,
        max_depth,
        buffer_size,
        max_rounds,
        rounds_per_batch,
        on_warmup,
        include_window_evidence,
        lower=True,
    )


def _prepare_stream(
    logdensity,
    jobs,
    data,
    schedule,
    diagonal,
    structured,
    acceptance,
    restart_dual,
    target_acceptance,
    max_depth,
    buffer_size,
    max_rounds,
    rounds_per_batch,
    on_warmup,
    include_window_evidence,
    *,
    lower,
):
    if on_warmup is not None and not callable(on_warmup):
        raise TypeError("on_warmup must be callable.")
    if not isinstance(include_window_evidence, (bool, np.bool_)):
        raise TypeError("include_window_evidence must be a boolean.")
    if include_window_evidence and on_warmup is None:
        raise ValueError("include_window_evidence requires an on_warmup callback.")
    max_depth = index(max_depth)
    if not 1 <= max_depth <= 30:
        raise ValueError("max_depth must be between 1 and 30.")
    if max_rounds is not None and index(max_rounds) < 0:
        raise ValueError("max_rounds must be nonnegative.")
    if rounds_per_batch is not None and index(rounds_per_batch) < 1:
        raise ValueError("rounds_per_batch must be positive.")
    if schedule is None:
        if (
            diagonal is not None
            or structured is not None
            or acceptance != "mean"
            or not restart_dual
            or target_acceptance != 0.8
        ):
            raise ValueError("A warmup schedule is required for adaptation settings.")
        target, queue = _prepare_queue(logdensity, jobs, data)
    else:
        target, queue, _, lengths, policy, schedule, _ = _prepare_warmup(
            logdensity,
            jobs,
            None,
            diagonal,
            schedule,
            acceptance,
            restart_dual,
            target_acceptance,
            data,
            None,
            max_depth,
            structured,
        )
    width, dim = queue.position.shape
    if min(width, dim) < 1:
        raise ValueError("Sampling requires a chain and a coordinate.")
    size = max(256, width) if buffer_size is None else index(buffer_size)
    if size < width:
        raise ValueError("The buffer must hold at least one simultaneous draw per chain.")
    if max_rounds == 0:
        return None if lower else iter(())

    dtype = queue.position.dtype

    def buffers(data, metric):
        state, chains = _empty(width, dim, dtype, max_depth, size, data, metric)
        stats = None
        if schedule is not None:
            stats = jax.vmap(partial(_initial, n_windows=len(lengths), policy=policy))(
                state.lanes.proposal.position,
                state.lanes.proposal.grad,
                chains.step_size,
                chains.inverse_mass,
            )
        return state, chains, stats

    if lower:
        state, chains, stats = jax.eval_shape(buffers, None, queue.inverse_mass)

        def abstract_data(row, width):
            traced = _broadcast_data.trace(row, width)
            return _output_shapes(traced, _committed_sharding(row))

        chains = chains._replace(data=_chain_data(data, width, abstract_data))
    else:
        state, chains, stats = buffers(data, queue.inverse_mass)

    take, ids = np.ones(width, bool), np.arange(width, dtype=np.int32)
    refill, loop = _compiled(target, max_depth)
    if schedule is None:
        inputs = (state, chains, take, ids, queue, data)
    else:
        refill, _ = _compiled_warmup(
            target, max_depth, policy, schedule, acceptance, restart_dual
        )
        inputs = (state.lanes, chains, stats, take, ids, queue, data)

    if lower:
        traced = refill.trace(*inputs)
        startup = traced.lower()
        # An actual refill inherits its dataset's devices. JAX's trace carries
        # the output partition specs but can omit the concrete device assignment.
        placement = next(
            (
                x.sharding
                for x in jax.tree.leaves(data)
                if _committed_sharding(x) is not None
            ),
            None,
        )
        outputs = _output_shapes(traced, placement, startup)
    else:
        outputs = refill(*inputs)

    del inputs
    deliver_events = None
    if schedule is None:
        state, chains = outputs
        step = loop
        round_args = (None, False)  # No draw quota or waiting jobs.
        n_windows = 0
    else:
        lanes, chains, stats, sampling_keys, valid = outputs
        if not lower and not np.asarray(valid).all():
            raise ValueError("Nonfinite initial log density or gradient.")
        state = state._replace(lanes=lanes)
        wrap_keys = jax.random.wrap_key_data
        keys = (
            _output_shapes(jax.jit(wrap_keys).trace(sampling_keys), sampling_keys.sharding)
            if lower
            else wrap_keys(sampling_keys)
        )
        budgets = jnp.asarray(lengths, jnp.int32)
        round_args = (stats, keys, budgets, jnp.asarray(target_acceptance, dtype))
        step = _mixed_loop(
            target,
            max_depth,
            policy,
            schedule,
            acceptance,
            restart_dual,
            on_warmup is not None,
            include_window_evidence,
        )
        n_windows = len(lengths)
        if on_warmup is not None:
            deliver_events = partial(
                _deliver_events,
                on_warmup,
                last=n_windows - 1,
                schedule=schedule,
                policy=policy,
                window_evidence=include_window_evidence,
            )

    if lower:
        limit = _round_limit(0, max_rounds, rounds_per_batch)
        return LoweredStream(startup, step.lower(state, chains, *round_args, limit, True))
    return _round_batches(
        state,
        chains,
        step,
        round_args,
        loop,
        n_windows,
        max_rounds,
        rounds_per_batch,
        deliver_events,
    )


def _committed_sharding(array):
    return array.sharding if isinstance(array, jax.Array) and array.committed else None


def _output_shapes(traced, placement, lowered=None):
    shapes = traced.out_info
    if placement is None:
        return shapes
    if isinstance(placement, jax.sharding.NamedSharding):
        if placement.mesh.auto_axes:
            # With automatic partitioning the compiler chooses output layouts.
            # Query the executable for them without passing it any buffers.
            lowered = traced.lower() if lowered is None else lowered
            formats = lowered.compile().output_formats
            return jax.tree.map(
                lambda shape, fmt: shape.update(sharding=fmt), shapes, formats
            )

        def place(shape):
            spec = getattr(shape.sharding, "spec", jax.sharding.PartitionSpec())
            return shape.update(sharding=placement.update(spec=spec))

        return jax.tree.map(place, shapes)
    return jax.tree.map(lambda shape: shape.update(sharding=placement), shapes)


def _round_limit(rounds, max_rounds, rounds_per_batch):
    limit = np.iinfo(np.int32).max
    if max_rounds is not None:
        limit = min(limit, max_rounds - rounds)
    if rounds_per_batch is not None:
        limit = min(limit, rounds_per_batch - rounds % rounds_per_batch)
    return jnp.int32(limit)


def _round_batches(
    state,
    chains,
    step,
    round_args,
    loop,
    n_windows,
    max_rounds,
    rounds_per_batch,
    deliver_events,
):
    width = state.lanes.active.size
    first = True
    rounds = 0
    progress = np.zeros(width, np.int64)
    frozen = np.zeros(width, bool)
    failed = np.zeros(width, bool)
    while max_rounds is None or rounds < max_rounds:
        new_failures = np.empty(0, dtype=np.intp)
        limit = _round_limit(rounds, max_rounds, rounds_per_batch)
        result = step(state, chains, *round_args, limit, first)
        if step is loop:
            state = result
            del result
            rounds += int(state.num_iterations)
        else:
            state, chains, stats, *diagnostics = result
            del result
            rounds += int(state.num_iterations)
            round_args = (stats, *round_args[1:])
            current_failures = np.asarray(stats.failed)
            new_failures = np.flatnonzero(current_failures & ~failed)
            failed = current_failures
            ready = np.asarray(stats.phase) >= n_windows
            newly_ready = ready & ~frozen
            if np.any(newly_ready):
                progress[newly_ready] = np.asarray(stats.evaluations)[newly_ready].sum(
                    axis=1, dtype=np.int64
                )
                frozen = ready
            if deliver_events is not None:
                events, resume = diagnostics
                deliver_events(events, rounds, failed)
                del events, diagnostics
                if bool(resume):
                    continue
            if (ready | failed).all():
                step, round_args = loop, (None, False)
            del stats
        count = int(state.num_records)
        if (
            count
            or len(new_failures)
            or rounds == max_rounds
            or (rounds_per_batch is not None and rounds % rounds_per_batch == 0)
        ):
            # Host arrays stay valid after the device buffer is donated to the next call.
            records = Record(*(_host_record(x, count) for x in state.records))
            yield DrawBatch(
                records, rounds, _completion_rounds(records, progress), new_failures
            )
            if count:
                first = False
        if len(new_failures) and failed.all():
            return
        state = state._replace(num_records=jnp.int32(0))
