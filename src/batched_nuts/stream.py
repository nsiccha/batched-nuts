"""Stream completed draws while every chain continues to advance."""

from collections.abc import Iterator
from functools import lru_cache, partial
from operator import index
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from batched_nuts.nuts import (
    Jobs,
    Record,
    _compiled,
    _empty,
    _prepare_queue,
    _write_records,
)
from batched_nuts.warmup import _compiled_warmup, _initial, _prepare_warmup, _warm_step


class DrawBatch(NamedTuple):
    """Host records in completion order and cumulative leapfrog rounds since start."""

    records: Record
    num_rounds: int
    completion_rounds: np.ndarray
    failed_jobs: np.ndarray


@lru_cache(maxsize=16)
def _mixed_loop(target, max_depth, diagonal, schedule, acceptance, restart_dual):
    def run(state, chains, stats, keys, budgets, target_accept, limit, first):
        state = state._replace(num_records=jnp.int32(0), num_iterations=jnp.int32(0))
        step = partial(
            _warm_step,
            target,
            max_depth,
            diagonal,
            schedule,
            acceptance,
            restart_dual,
            budgets,
            target_accept,
            stream=True,
        )
        width = state.lanes.active.size
        capacity = state.records.job.size
        failed_before = stats.failed

        def go_on(carry):
            state, _, stats = carry
            return (
                (state.num_iterations < limit)
                & (state.num_records + width <= capacity)
                & (~first | (state.num_records == 0))
                & jnp.any(state.lanes.active & (stats.phase < budgets.size))
                & ~jnp.any(stats.failed & ~failed_before)
            )

        def advance(carry):
            state, chains, stats = carry
            chains, lanes, stats, records, done = jax.vmap(
                lambda chain, lane, adapt, key: step(chain, lane, adapt, sampling_key=key)
            )(chains, state.lanes, stats, keys)
            state = state._replace(
                lanes=lanes,
                records=_write_records(state, records, done),
                num_records=state.num_records + jnp.sum(done, dtype=jnp.int32),
                num_iterations=state.num_iterations + 1,
            )
            return state, chains, stats

        return jax.lax.while_loop(go_on, advance, (state, chains, stats))

    return jax.jit(run, donate_argnums=(0, 1, 2))


def _host_record(column, count):
    if count == 0:
        return np.empty((0, *column.shape[1:]), column.dtype)
    if count < column.shape[0] // 4:
        # Early draws can be sparse in thousands of lanes. Limit transfers to
        # power-of-two shapes so the number of distinct slices stays logarithmic.
        column = column[: 1 << (count - 1).bit_length()]
    return np.asarray(column)[:count].copy()


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
    acceptance="mean",
    restart_dual=True,
    target_acceptance=0.8,
    max_depth=10,
    buffer_size=None,
    max_rounds=None,
    rounds_per_batch=None,
) -> Iterator[DrawBatch]:
    """Run one lane per job. Yield bounded host batches without a per-chain draw quota.

    With a warmup schedule, each chain begins sampling when its own adaptation ends.
    With fitted jobs, omit the schedule. The first completed draw is yielded during
    the round that produced it (possibly with simultaneous draws). Later transfers
    use the record buffer. Set rounds_per_batch to also yield at round checkpoints,
    including empty batches during warmup. Set max_rounds for an exact finite budget,
    or close the iterator to stop. Both limits count leapfrogs, including warmup.
    Each record has its exact trajectory-completion round in completion_rounds.
    The initial batched evaluation of the starting positions precedes round one.
    If a job exhausts its adaptive windows, its ID is yielded once in failed_jobs.
    Other chains continue. The iterator ends if every job exhausts its windows.
    """
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
            or acceptance != "mean"
            or not restart_dual
            or target_acceptance != 0.8
        ):
            raise ValueError("A warmup schedule is required for adaptation settings.")
        target, queue = _prepare_queue(logdensity, jobs, data)
    else:
        target, queue, _, lengths, diagonal, schedule, _ = _prepare_warmup(
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
        )
    width, dim = queue.position.shape
    if min(width, dim) < 1:
        raise ValueError("Sampling requires a chain and a coordinate.")
    size = max(256, width) if buffer_size is None else index(buffer_size)
    if size < width:
        raise ValueError("The buffer must hold at least one simultaneous draw per chain.")
    if max_rounds == 0:
        return
    dtype = queue.position.dtype
    state, chains = _empty(width, dim, dtype, max_depth, size, data)
    take, ids = np.ones(width, bool), np.arange(width, dtype=np.int32)
    refill, loop = _compiled(target, max_depth)
    stats = None
    if schedule is None:
        state, chains = refill(state, chains, take, ids, queue, data)
    else:
        budgets = jnp.asarray(lengths, jnp.int32)
        accept = jnp.asarray(target_acceptance, dtype)
        stats = jax.vmap(partial(_initial, n_windows=len(lengths)))(
            state.lanes.proposal.position,
            state.lanes.proposal.grad,
            chains.step_size,
            chains.inverse_mass,
        )
        init, _ = _compiled_warmup(
            target, max_depth, diagonal, schedule, acceptance, restart_dual
        )
        lanes, chains, stats, sampling_keys, valid = init(
            state.lanes, chains, stats, take, ids, queue, data
        )
        if not np.asarray(valid).all():
            raise ValueError("Nonfinite initial log density or gradient.")
        state = state._replace(lanes=lanes)
        del lanes
        keys = jax.random.wrap_key_data(sampling_keys)
        mixed = _mixed_loop(target, max_depth, diagonal, schedule, acceptance, restart_dual)
    first = True
    rounds = 0
    progress = np.zeros(width, np.int64)
    frozen = np.zeros(width, bool)
    failed = np.zeros(width, bool)
    while max_rounds is None or rounds < max_rounds:
        new_failures = np.empty(0, dtype=np.intp)
        limit = np.iinfo(np.int32).max
        if max_rounds is not None:
            limit = min(limit, max_rounds - rounds)
        if rounds_per_batch is not None:
            limit = min(limit, rounds_per_batch - rounds % rounds_per_batch)
        if stats is None:
            state = loop(state, chains, None, False, jnp.int32(limit), first)
        else:
            state, chains, stats = mixed(
                state, chains, stats, keys, budgets, accept, jnp.int32(limit), first
            )
            current_failures = np.asarray(stats.failed)
            new_failures = np.flatnonzero(current_failures & ~failed)
            failed = current_failures
            ready = np.asarray(stats.phase) >= len(lengths)
            newly_ready = ready & ~frozen
            if np.any(newly_ready):
                progress[newly_ready] = np.asarray(stats.evaluations)[newly_ready].sum(
                    axis=1, dtype=np.int64
                )
                frozen = ready
            if (ready | failed).all():
                stats = None
                del keys, sampling_keys
        rounds += int(state.num_iterations)
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
