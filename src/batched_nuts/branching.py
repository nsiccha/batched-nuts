"""Branch a streaming run from completed transitions of its original chains."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from itertools import islice
from operator import index
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from batched_nuts.nuts import _start_lanes

if TYPE_CHECKING:
    from batched_nuts.stream import DrawBatch


class Parent(NamedTuple):
    """Available source rounds for one frozen original chain on its target dataset."""

    job: int
    dataset: int
    draw_round: int | None
    warmup_rounds: tuple[int, ...]


class Branch(NamedTuple):
    """Request a child from an advertised round. Settle counts discarded transitions."""

    parent: int
    source_round: int
    settle: int | None = None


class BranchOrigin(NamedTuple):
    """Lineage for a newly admitted child, with rounds on the stream's clock."""

    job: int
    parent: int
    dataset: int
    source_round: int
    warmup: bool
    admission_round: int
    settle: int


@dataclass(frozen=True)
class Branching:
    """Call choose(batch, eligible_parents, remaining_width) to request children."""

    key: jax.Array
    ready_target: int
    max_chains: int
    choose: Callable[["DrawBatch", tuple[Parent, ...], int], Iterable[Branch] | None]
    warmup_history: int = 0


class _History(NamedTuple):
    """The last H completed warmup transitions per original chain, on device."""

    positions: jax.Array
    lengths: jax.Array
    next_slot: jax.Array

    @classmethod
    def empty(cls, roots, capacity, dim, dtype):
        return cls(
            jnp.zeros((roots, capacity, dim), dtype),
            jnp.zeros((roots, capacity), jnp.int32),
            jnp.zeros(roots, jnp.int32),
        )

    def record(self, done, records):
        roots, capacity = self.lengths.shape
        finished = done[:roots]
        # An out-of-bounds slot on unfinished rows drops that write.
        row = jnp.arange(roots)
        col = jnp.where(finished, self.next_slot, capacity)
        return _History(
            self.positions.at[row, col].set(records.position[:roots], mode="drop"),
            self.lengths.at[row, col].set(
                records.num_integration_steps[:roots], mode="drop"
            ),
            jnp.where(finished, (self.next_slot + 1) % capacity, self.next_slot),
        )


@partial(jax.jit, static_argnums=0)
def _grow(target, lanes, chains, stats, keys, parents, positions, key_words):
    """Initialize new rows with frozen parent tuning, then concatenate the states."""

    def append(existing, extra):
        return jax.tree.map(lambda a, b: jnp.concatenate((a, b)), existing, extra)

    def select(tree):
        return jax.tree.map(lambda x: x[parents], tree)

    child_jobs = chains.job.size + jnp.arange(parents.size, dtype=jnp.int32)
    seed = jax.random.wrap_key_data(key_words)
    new_keys = jax.vmap(partial(jax.random.fold_in, seed))(child_jobs)
    children = select(chains)._replace(job=child_jobs, key=new_keys)
    starts = _start_lanes(target, select(lanes), children, positions)
    if stats is not None:
        # A source is frozen. Its terminal phase makes the mixed step advance the
        # child with its fitted tuning while other original chains are still warming.
        stats = append(stats, select(stats))
        keys = append(keys, new_keys)
    return append(lanes, starts), append(chains, children), stats, keys


class _Branches:
    def __init__(self, policy, datasets, buffer_size, target):
        self.target = target
        self.policy = policy
        self.datasets = np.asarray(datasets)
        self.width = len(datasets)
        if not 1 <= index(policy.ready_target) <= self.width:
            raise ValueError("ready_target must be between one and the initial width.")
        if not self.width <= index(policy.max_chains) <= np.iinfo(np.int32).max:
            raise ValueError("max_chains must cover the initial width and fit in int32.")
        if buffer_size is not None and index(buffer_size) < policy.max_chains:
            raise ValueError("An explicit buffer_size must cover max_chains.")
        if not callable(policy.choose):
            raise TypeError("choose must be callable.")
        if index(policy.warmup_history) < 0:
            raise ValueError("warmup_history must be nonnegative.")
        # Derive child keys on the same device as the chains at admission. The
        # caller can supply a key committed to a different device.
        self.key_words = np.asarray(jax.random.key_data(policy.key))
        if jax.random.wrap_key_data(self.key_words).shape:
            raise ValueError("Branching requires one random key.")
        self.history = [{} for _ in datasets]
        self.latest = [None] * self.width
        self.settling = np.zeros(self.width, np.int64)

    @property
    def room(self):
        return self.policy.max_chains - self.width

    def save_frozen(self, history, newly_ready, round_number):
        roots = np.flatnonzero(newly_ready[: len(self.datasets)])
        if not len(roots) or self.width == self.policy.max_chains:
            return
        positions, lengths, heads = jax.device_get(
            jax.tree.map(lambda x: x[roots], history)
        )
        for root, points, steps, head in zip(roots, positions, lengths, heads, strict=True):
            order = (np.arange(len(steps)) + head) % len(steps)
            order = order[steps[order] > 0]
            cumulative = np.cumsum(steps[order], dtype=np.int64)
            times = round_number - cumulative[-1] + cumulative
            self.history[root] = {
                int(t): points[i].copy() for t, i in zip(times, order, strict=True)
            }

    def retained(self, batch):
        records = batch.records
        discard = self.settling[records.job]
        if np.any(discard):
            keep = records.draw >= discard
            records = type(records)(*(x[keep] for x in records))
            records = records._replace(
                draw=records.draw - discard[keep].astype(records.draw.dtype)
            )
            batch = batch._replace(
                records=records, completion_rounds=batch.completion_rounds[keep]
            )
        return batch

    def choose(self, batch, frozen):
        room = self.room
        if not room:
            return (), None
        ids, reverse_indices = np.unique(batch.records.job[::-1], return_index=True)
        for job, offset in zip(ids, reverse_indices, strict=True):
            if job < len(self.latest):
                i = -1 - offset
                self.latest[job] = (
                    int(batch.completion_rounds[i]),
                    batch.records.position[i].copy(),
                )

        ready = np.flatnonzero(frozen[: len(self.datasets)])
        groups, counts = np.unique(self.datasets[ready], return_counts=True)
        eligible = ready[
            np.isin(self.datasets[ready], groups[counts >= self.policy.ready_target])
        ]
        if not len(eligible):
            return (), None
        parents = tuple(
            Parent(
                int(i),
                int(self.datasets[i]),
                self.latest[i][0] if self.latest[i] is not None else None,
                tuple(self.history[i]),
            )
            for i in eligible
        )
        requests = self.policy.choose(batch, parents, room)
        if requests is None:
            return (), None
        requests = tuple(islice(requests, room + 1))
        if len(requests) > room:
            raise ValueError("Branch requests exceed max_chains.")
        allowed = set(eligible)
        origins, points = [], []
        for i, req in enumerate(requests):
            parent, source = index(req.parent), index(req.source_round)
            if parent not in allowed:
                raise ValueError("Branch parent must be an eligible frozen original chain.")
            latest = self.latest[parent]
            warmup = latest is None or source != latest[0]
            if warmup:
                point = self.history[parent].get(source)
                if point is None:
                    raise ValueError("Source round is not available for this parent.")
                if req.settle is None:
                    raise ValueError("Choose a settling period for a warmup source.")
            else:
                point = latest[1]
            settle = 0 if req.settle is None else index(req.settle)
            if not 0 <= settle <= np.iinfo(np.int32).max:
                raise ValueError("settle must be a nonnegative int32 transition count.")
            origins.append(
                BranchOrigin(
                    self.width + i,
                    parent,
                    int(self.datasets[parent]),
                    source,
                    warmup,
                    batch.num_rounds,
                    settle,
                )
            )
            points.append(point)
        return tuple(origins), np.stack(points) if points else None

    def grow(self, state, chains, stats, keys, origins, points):
        parents = np.array([b.parent for b in origins], np.int32)
        lanes, chains, stats, keys = _grow(
            self.target, state.lanes, chains, stats, keys, parents, points, self.key_words
        )
        self.width += len(origins)
        self.settling = np.r_[self.settling, [b.settle for b in origins]]
        if self.width == self.policy.max_chains:
            self.history.clear()
            self.latest.clear()
        extra = self.width - state.records.job.size
        if extra > 0:

            def extend(x):
                return jnp.concatenate((x, jnp.zeros((extra, *x.shape[1:]), x.dtype)))

            state = state._replace(records=jax.tree.map(extend, state.records))
        return state._replace(lanes=lanes), chains, stats, keys
