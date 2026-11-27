"""Stream completed draws while every chain continues to advance."""

from collections.abc import Callable, Iterator
from functools import lru_cache, partial
from operator import index
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from batched_nuts.branching import Branching, BranchOrigin, _Branches, _History
from batched_nuts.nuts import (
    Jobs,
    Record,
    _bind_shared_data,
    _broadcast_data,
    _chain_data,
    _compiled,
    _empty,
    _prepare_data,
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


class BranchBatch(NamedTuple):
    """Draws and any children admitted after the last round in this batch."""

    records: Record
    num_rounds: int
    completion_rounds: np.ndarray
    failed_jobs: np.ndarray
    branches: tuple[BranchOrigin, ...]


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
    save_history=False,
    stop_at_freeze=False,
):
    def run(
        state,
        chains,
        shared_data,
        stats,
        keys,
        budgets,
        target_accept,
        limit,
        first,
        history=None,
    ):
        state = state._replace(num_iterations=jnp.int32(0))
        if not observe:
            state = state._replace(num_records=jnp.int32(0))
        step = partial(
            _warm_step,
            _bind_shared_data(target, shared_data),
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
            save_history=save_history,
        )
        width = state.lanes.active.size
        capacity = state.records.job.size
        failed_before = stats.failed

        def go_on(carry):
            state, _, stats, _, _ = carry
            return (
                (state.num_iterations < limit)
                & (state.num_records + width <= capacity)
                & (~first | (state.num_records == 0))
                & jnp.any(state.lanes.active & (stats.phase < budgets.size))
                & ~jnp.any(stats.failed & ~failed_before)
            )

        def advance(carry):
            state, chains, stats, _, history = carry
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
            if save_history:
                history = history.record(result[-1], records)
            return state, chains, stats, events, history

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
                events = carry[3]
                running &= ~jnp.any(events.closed | events.divergent)
            elif stop_at_freeze:
                ready = carry[2].phase >= budgets.size
                running &= ~jnp.any(ready & (stats.phase < budgets.size))
            return running

        carry = jax.lax.while_loop(
            condition, advance, (state, chains, stats, initial, history)
        )
        # An event-only return leaves buffered draws for their original yield point.
        result = (*carry[:4], go_on(carry)) if observe else carry[:3]
        return (*result, carry[4]) if save_history else result

    return jax.jit(run, donate_argnames=("state", "chains", "stats", "history"))


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
    branching: Branching | None = None,
) -> Iterator[DrawBatch | BranchBatch]:
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
    With branching, yield BranchBatch objects. Call branching.choose at eligible
    freeze, draw and checkpoint batches to admit children with frozen parent tuning.

    Pass `data=observations` and omit `jobs.dataset` to share one runtime
    dataset across all chains. With explicit indices, the density receives each
    job's selected row of the data tree. Equal indices also share one row.
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
        branching,
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
    branching: Branching | None = None,
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
        branching,
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
    branching,
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
    if branching is not None and not isinstance(branching, Branching):
        raise TypeError("branching must be a Branching policy.")
    branches = None
    if branching is not None:
        datasets = np.zeros(width, np.int32) if queue.dataset is None else queue.dataset
        branches = _Branches(branching, datasets, buffer_size, target)
    if max_rounds == 0:
        return None if lower else iter(())

    data, shared_data = _prepare_data(data, queue, lower=lower)
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
        inputs = (state, chains, take, ids, queue, data, shared_data)
    else:
        refill, _ = _compiled_warmup(
            target, max_depth, policy, schedule, acceptance, restart_dual
        )
        inputs = (state.lanes, chains, stats, take, ids, queue, data, shared_data)

    if lower:
        traced = refill.trace(*inputs)
        startup = traced.lower()
        # An actual refill inherits its dataset's devices. JAX's trace carries
        # the output partition specs but can omit the concrete device assignment.
        placement = next(
            (
                x.sharding
                for x in jax.tree.leaves((data, shared_data))
                if _committed_sharding(x) is not None
            ),
            None,
        )
        outputs = _output_shapes(traced, placement, startup)
    else:
        outputs = refill(*inputs)

    del inputs
    deliver_events = mixed_kernel = None
    history_args = ()
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
        mixed_kernel = partial(
            _mixed_loop,
            target,
            max_depth,
            policy,
            schedule,
            acceptance,
            restart_dual,
            on_warmup is not None,
            include_window_evidence,
        )
        can_grow = branching is not None and width < branching.max_chains
        if can_grow and branching.warmup_history:
            make_history = partial(
                _History.empty, width, branching.warmup_history, dim, dtype
            )
            history_args = (jax.eval_shape(make_history) if lower else make_history(),)
        step = mixed_kernel(bool(history_args), can_grow)
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
        return LoweredStream(
            startup,
            step.lower(state, chains, shared_data, *round_args, limit, True, *history_args),
        )
    return _round_batches(
        state,
        chains,
        shared_data,
        step,
        round_args,
        loop,
        n_windows,
        max_rounds,
        rounds_per_batch,
        deliver_events,
        branches,
        mixed_kernel,
        history_args,
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
    shared_data,
    step,
    round_args,
    loop,
    n_windows,
    max_rounds,
    rounds_per_batch,
    deliver_events,
    branches,
    mixed_kernel,
    history_args,
):
    width = state.lanes.active.size
    first = True
    rounds = 0
    progress = np.zeros(width, np.int64)
    frozen = np.full(width, n_windows == 0, bool)
    failed = np.zeros(width, bool)
    while max_rounds is None or rounds < max_rounds:
        freeze_checkpoint = False
        new_failures = np.empty(0, dtype=np.intp)
        limit = _round_limit(rounds, max_rounds, rounds_per_batch)
        result = step(state, chains, shared_data, *round_args, limit, first, *history_args)
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
            if history_args:
                history_args = (diagnostics.pop(),)
                branches.save_frozen(history_args[0], newly_ready, rounds)
            freeze_checkpoint = (
                branches is not None and branches.room > 0 and bool(np.any(newly_ready))
            )
            if deliver_events is not None:
                events, resume = diagnostics
                deliver_events(events, rounds, failed)
                del events, diagnostics
                if bool(resume) and not freeze_checkpoint:
                    continue
            if (ready | failed).all():
                step, round_args = loop, (None, False)
                history_args = ()
            del stats
        count = int(state.num_records)
        if (
            count
            or freeze_checkpoint
            or len(new_failures)
            or rounds == max_rounds
            or (rounds_per_batch is not None and rounds % rounds_per_batch == 0)
        ):
            # Host arrays stay valid after the device buffer is donated to the next call.
            records = Record(*(_host_record(x, count) for x in state.records))
            batch = DrawBatch(
                records, rounds, _completion_rounds(records, progress), new_failures
            )
            if branches is None:
                yield batch
            else:
                batch = branches.retained(batch)
                origins = ()
                if rounds != max_rounds:
                    origins, points = branches.choose(batch, frozen)
                    if origins:
                        stats, keys = round_args[:2] if step is not loop else (None, None)
                        state, chains, stats, keys = branches.grow(
                            state, chains, stats, keys, origins, points, shared_data
                        )
                        if step is not loop:
                            round_args = (stats, keys, *round_args[2:])
                            if not branches.room:
                                history_args = ()
                                step = mixed_kernel()
                        del stats, keys, points
                        progress = np.r_[progress, np.full(len(origins), rounds, np.int64)]
                        frozen = np.r_[frozen, np.ones(len(origins), bool)]
                        failed = np.r_[failed, np.zeros(len(origins), bool)]
                yield BranchBatch(*batch, origins)
            if len(batch.records.job):
                first = False
        if len(new_failures) and failed.all():
            return
        state = state._replace(num_records=jnp.int32(0))
