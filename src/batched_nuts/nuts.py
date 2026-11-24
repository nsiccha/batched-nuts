"""NUTS for many chains at once, batched by gradient evaluation.

The chains run side by side in the *lanes* of one `jax.vmap`.  Every iteration of the
batched loop evaluates the gradient exactly once per lane: one leapfrog step.  What that
step means differs from lane to lane.  It may extend a trajectory, end a subtree or end a
transition, in which case the lane records its draw and begins the next transition in the
following iteration.  No lane ever waits for another.  A lane whose chain has all its
draws is handed the next chain from a queue.

The transition is the one of `blackjax.nuts`: multinomial sampling, biased progressive
sampling between subtrees, uniform progressive sampling within a subtree, generalized
U-turn checks of every balanced block of a subtree and of the whole trajectory after each
doubling, and a divergence when the energy error exceeds 1000.  The random numbers are
derived from the key of a transition in the same way and the leapfrog arithmetic is
written in the same order, so the same key gives the same trajectory and the same draw
(up to how the compiler rounds the arithmetic of the log density itself).
"""

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

DIVERGENCE_THRESHOLD = 1000


class Jobs(NamedTuple):
    """The chains to run, one row per chain."""

    key: Any  # (num_jobs,) PRNG keys; transition t of a chain uses `fold_in(key, t)`
    position: Any  # (num_jobs, dim) initial positions
    step_size: Any  # (num_jobs,)
    inverse_mass: Any  # (num_jobs, dim) diagonal of the inverse mass matrix
    dataset: Any = None  # (num_jobs,) which slice of `data` the chain's density reads


class Draws(NamedTuple):
    """The result of `sample`.  Arrays have the leading axes (num_jobs, num_draws)."""

    position: np.ndarray
    logdensity: np.ndarray
    num_integration_steps: np.ndarray
    is_divergent: np.ndarray
    is_turning: np.ndarray
    acceptance_rate: np.ndarray
    num_iterations: int  # iterations of the batched loop, one gradient per lane each
    num_evaluations: int  # those of the gradients that advanced a chain


class Chain(NamedTuple):
    """What a lane is running.  Constant while the batched loop runs."""

    job: Any  # row of the chain in `Jobs`
    key: Any
    step_size: Any
    inverse_mass: Any
    data: Any  # the chain's slice of the data


class Point(NamedTuple):
    """One end of a trajectory."""

    position: Any
    momentum: Any
    grad: Any


class Proposal(NamedTuple):
    """A candidate for the next draw, chosen among the states of (part of) a trajectory."""

    position: Any
    logdensity: Any
    grad: Any
    weight: Any  # log of the sum of exp(initial energy - energy) over those states
    sum_log_accept: Any  # log of the sum of min(1, exp(initial energy - energy))


class Lane(NamedTuple):
    """Where a lane is in its chain: what it carries from one iteration to the next."""

    # Whether the lane has a chain to run.  One without keeps running its last chain,
    # which keeps its values valid, but records nothing.
    active: Any
    draw: Any  # index of the current transition = number of draws recorded so far
    # The state of the chain.  During a transition it doubles as the proposal chosen
    # among the trajectory built so far, which starts out as the state of the chain.
    proposal: Proposal
    # The current transition.
    tree_key: Any
    energy: Any  # of the initial state
    depth: Any  # number of completed doublings; the current subtree has 2**depth leaves
    direction: Any  # +1 or -1: the end of the trajectory the current subtree extends
    left: Point
    right: Point
    momentum_sum: Any  # over the trajectory, without the current subtree
    num_steps: Any  # leapfrog steps of the trajectory, without the current subtree
    # The current subtree.
    leaf_keys: Any  # leaf number i of the subtree uses `fold_in(leaf_keys, i)`
    accept_key: Any  # decides whether the subtree's proposal replaces the trajectory's
    leaf: Any  # index of the next leaf
    sub: Proposal
    sub_momentum_sum: Any
    checkpoint_momentum: Any  # (max_depth, dim), see `_checkpoint_rows`
    checkpoint_sum: Any  # (max_depth, dim)


class Record(NamedTuple):
    """A finished transition."""

    job: Any
    draw: Any
    position: Any
    logdensity: Any
    num_integration_steps: Any
    is_divergent: Any
    is_turning: Any
    acceptance_rate: Any


class State(NamedTuple):
    """The carry of the batched loop."""

    lanes: Lane  # every field with a leading axis (num_lanes,)
    records: Record  # every field with a leading axis (buffer_size,)
    num_records: Any  # rows of `records` that are filled
    num_iterations: Any  # since the loop was entered
    num_evaluations: Any  # since the loop was entered


def _where(condition, a, b):
    return jax.tree.map(lambda x, y: jnp.where(condition, x, y), a, b)


def _kinetic_energy(inverse_mass, momentum):
    return 0.5 * jnp.dot(inverse_mass * momentum, momentum)


def _is_turning(inverse_mass, momentum_left, momentum_right, momentum_sum):
    """The generalized U-turn criterion for a stretch of trajectory (or a stack of them)."""
    rho = momentum_sum - (momentum_right + momentum_left) / 2
    turning_left = jnp.sum(inverse_mass * momentum_left * rho, axis=-1) <= 0
    turning_right = jnp.sum(inverse_mass * momentum_right * rho, axis=-1) <= 0
    return turning_left | turning_right


def _checkpoint_rows(leaf):
    """Rows of the checkpoint table to compare with after leaf number `leaf`.

    A subtree is checked for U-turns block by block: every balanced block of 2, 4, 8, ...
    consecutive leaves must not turn.  The blocks that end at an odd leaf start at even
    leaves, and it is enough to remember max_depth of those: an even leaf stores its
    momentum and the momentum sum up to it in row `last` (the number of blocks still
    open before it), and an odd leaf closes the blocks stored in rows `first..last`.
    An even leaf closes no block (`first > last`).
    """
    last = jax.lax.population_count(leaf >> 1)
    num_closing = jax.lax.population_count((~leaf & (leaf + 1)) - 1)
    return last - num_closing + 1, last


def _begin_transition(chain: Chain, lane: Lane) -> Lane:
    """Draw a momentum and start a trajectory at the state of the chain."""
    state = lane.proposal
    dtype = state.position.dtype
    momentum_key, tree_key = jax.random.split(jax.random.fold_in(chain.key, lane.draw), 2)
    noise = jax.random.normal(momentum_key, state.position.shape, dtype)
    momentum = (1 / jnp.sqrt(chain.inverse_mass)) * noise
    end = Point(state.position, momentum, state.grad)
    return lane._replace(
        proposal=state._replace(
            weight=jnp.zeros((), dtype), sum_log_accept=jnp.full((), -jnp.inf, dtype)
        ),
        tree_key=tree_key,
        energy=-state.logdensity + _kinetic_energy(chain.inverse_mass, momentum),
        depth=jnp.int32(0),
        left=end,
        right=end,
        momentum_sum=momentum,
        num_steps=jnp.int32(0),
    )


def _begin_doubling(lane: Lane) -> Lane:
    """Start a subtree: choose the end of the trajectory it extends, derive its keys."""
    keys = jax.random.split(jax.random.fold_in(lane.tree_key, lane.depth), 3)
    direction = jnp.where(jax.random.bernoulli(keys[0]), 1, -1).astype(jnp.int32)
    return lane._replace(
        direction=direction, leaf_keys=keys[1], accept_key=keys[2], leaf=jnp.int32(0)
    )


def _advance(logdensity, max_depth, chain: Chain, lane: Lane, *, warmup_info=False):
    """Take a leapfrog step, exposing the two proposal probabilities for warmup."""
    inverse_mass = chain.inverse_mass

    # One leapfrog step from the end of the trajectory that the current subtree extends.
    forward = lane.direction > 0
    start = _where(forward, lane.right, lane.left)
    step_size = lane.direction * chain.step_size
    momentum = start.momentum + step_size * 0.5 * start.grad
    position = start.position + step_size * (inverse_mass * momentum)
    logp, grad = jax.value_and_grad(logdensity)(position, chain.data)
    momentum = momentum + step_size * 0.5 * grad
    end = Point(position, momentum, grad)

    # The new state as a proposal of its own.
    delta = lane.energy - (-logp + _kinetic_energy(inverse_mass, momentum))
    delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
    leaf = Proposal(position, logp, grad, delta, jnp.minimum(delta, 0.0))
    diverging = -delta > DIVERGENCE_THRESHOLD

    # Within the subtree every state is equally entitled to be the proposal (uniform
    # progressive sampling): the new one replaces the old one with probability
    # weight / (weight of all states of the subtree so far).
    first = lane.leaf == 0
    old = lane.sub
    leaf_probability = jax.scipy.special.expit(leaf.weight - old.weight)
    accept = first | jax.random.bernoulli(
        jax.random.fold_in(lane.leaf_keys, lane.leaf), leaf_probability
    )
    sub = _where(accept, leaf, old)._replace(
        weight=jnp.where(first, leaf.weight, jnp.logaddexp(old.weight, leaf.weight)),
        sum_log_accept=jnp.where(
            first,
            leaf.sum_log_accept,
            jnp.logaddexp(old.sum_log_accept, leaf.sum_log_accept),
        ),
    )
    sub_momentum_sum = jnp.where(first, momentum, lane.sub_momentum_sum + momentum)

    # Does a balanced block of the subtree that ends at this leaf make a U-turn?
    first_row, last_row = _checkpoint_rows(lane.leaf)
    even = lane.leaf % 2 == 0
    checkpoint_momentum = lane.checkpoint_momentum.at[last_row].set(
        jnp.where(even, momentum, lane.checkpoint_momentum[last_row])
    )
    checkpoint_sum = lane.checkpoint_sum.at[last_row].set(
        jnp.where(even, sub_momentum_sum, lane.checkpoint_sum[last_row])
    )
    block_sum = sub_momentum_sum - checkpoint_sum + checkpoint_momentum
    block_turning = _is_turning(inverse_mass, checkpoint_momentum, momentum, block_sum)
    rows = jnp.arange(max_depth)
    sub_turning = jnp.any(block_turning & (first_row <= rows) & (rows <= last_row))

    num_leaves = lane.leaf + 1
    sub_done = (num_leaves == 1 << lane.depth) | sub_turning | diverging
    left = _where(forward, lane.left, end)
    right = _where(forward, end, lane.right)

    # At the end of a subtree: merge it into the trajectory.  A complete subtree hands
    # over its proposal with probability min(1, its weight / weight of the trajectory)
    # (biased progressive sampling); one that stopped early is not sampled from.
    top = lane.proposal
    complete = ~(sub_turning | diverging)
    subtree_probability = jnp.clip(jnp.exp(sub.weight - top.weight), max=1)
    take = complete & jax.random.bernoulli(lane.accept_key, subtree_probability)
    merged = _where(take, sub, top)._replace(
        weight=jnp.where(complete, jnp.logaddexp(top.weight, sub.weight), top.weight),
        sum_log_accept=jnp.logaddexp(top.sum_log_accept, sub.sum_log_accept),
    )
    momentum_sum = lane.momentum_sum + sub_momentum_sum
    num_steps = lane.num_steps + num_leaves
    turning = sub_turning | _is_turning(
        inverse_mass, left.momentum, right.momentum, momentum_sum
    )
    depth = lane.depth + 1
    ended = sub_done & (diverging | turning | (depth == max_depth))

    record = Record(
        job=chain.job,
        draw=lane.draw,
        position=merged.position,
        logdensity=merged.logdensity,
        num_integration_steps=num_steps,
        is_divergent=diverging,
        is_turning=turning,
        acceptance_rate=jnp.exp(merged.sum_log_accept) / num_steps,
    )
    done = lane.active & ended

    lane = lane._replace(
        draw=lane.draw + done,
        proposal=_where(sub_done, merged, top),
        depth=jnp.where(sub_done, depth, lane.depth),
        left=left,
        right=right,
        momentum_sum=jnp.where(sub_done, momentum_sum, lane.momentum_sum),
        num_steps=jnp.where(sub_done, num_steps, lane.num_steps),
        leaf=num_leaves,
        sub=sub,
        sub_momentum_sum=sub_momentum_sum,
        checkpoint_momentum=checkpoint_momentum,
        checkpoint_sum=checkpoint_sum,
    )
    probabilities = (
        jnp.where(first, 1.0, leaf_probability),
        jnp.where(sub_done & complete, subtree_probability, 0.0),
    )
    result = lane, record, ended, sub_done, end, probabilities
    return (*result, delta, accept, sub_done & take) if warmup_info else result


def _begin_next(chain, lane, ended, sub_done):
    lane = _where(ended, _begin_transition(chain, lane), lane)
    return _where(sub_done, _begin_doubling(lane), lane)


def _step(logdensity, max_depth, chain, lane):
    lane, record, ended, sub_done, _, _ = _advance(logdensity, max_depth, chain, lane)
    return _begin_next(chain, lane, ended, sub_done), record, lane.active & ended


def _write_records(state, records, done):
    buffer_size = state.records.job.shape[0]
    row = state.num_records + jnp.cumsum(done, dtype=jnp.int32) - 1
    row = jnp.where(done, row, buffer_size)  # out of range: not written
    return jax.tree.map(
        lambda buffer, new: buffer.at[row].set(new, mode="drop"), state.records, records
    )


def _iteration(logdensity, max_depth, chains: Chain, num_draws, state: State) -> State:
    """One gradient evaluation in every lane; collect the transitions that finished."""
    count = partial(jnp.sum, dtype=jnp.int32)
    num_evaluations = state.num_evaluations + count(state.lanes.active)
    lanes, records, done = jax.vmap(partial(_step, logdensity, max_depth))(
        chains, state.lanes
    )
    records = _write_records(state, records, done)
    finished = jnp.bool_(False) if num_draws is None else done & (lanes.draw == num_draws)
    return State(
        lanes=lanes._replace(active=lanes.active & ~finished),
        records=records,
        num_records=state.num_records + count(done),
        num_iterations=state.num_iterations + 1,
        num_evaluations=num_evaluations,
    )


def _loop(
    logdensity, max_depth, state: State, chains: Chain, num_draws, chains_waiting,
    max_rounds=None, first_draw=False,
):
    """Iterate until the host has something to do.

    That is: all chains are done, or a lane is idle while chains are waiting, or the
    record buffer has no room for another iteration.
    """
    num_lanes = state.lanes.active.shape[0]
    buffer_size = state.records.job.shape[0]
    zero = jnp.int32(0)
    state = state._replace(num_iterations=zero, num_evaluations=zero)

    def go_on(state):
        active = state.lanes.active
        room = state.num_records + num_lanes <= buffer_size
        budget = True if max_rounds is None else state.num_iterations < max_rounds
        emit = ~jnp.asarray(first_draw) | (state.num_records == 0)
        return emit & budget & room & jnp.any(active) & ~(chains_waiting & jnp.any(~active))

    iteration = partial(_iteration, logdensity, max_depth, chains, num_draws)
    return jax.lax.while_loop(go_on, iteration, state)


def _assign(logdensity, lanes: Lane, chains: Chain, take, job, jobs: Jobs, data):
    """Hand the chains in the rows of `jobs` to the lanes marked in `take`.

    `take`, `job` and `jobs` have one row per lane; the rows of lanes that take nothing
    are ignored.  The new chains start at their initial positions, which are evaluated
    here, and begin their first transition.
    """

    def choose(new, old):
        return jnp.where(take.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)

    chains = Chain(
        job=choose(job, chains.job),
        key=choose(jax.random.wrap_key_data(jobs.key), chains.key),
        step_size=choose(jobs.step_size, chains.step_size),
        inverse_mass=choose(jobs.inverse_mass, chains.inverse_mass),
        data=jax.tree.map(
            lambda table, old: choose(table[jobs.dataset], old), data, chains.data
        ),
    )
    logp, grad = jax.vmap(jax.value_and_grad(logdensity))(jobs.position, chains.data)
    taken = lanes._replace(
        active=jnp.ones_like(lanes.active),
        draw=jnp.zeros_like(lanes.draw),
        proposal=lanes.proposal._replace(
            position=jobs.position, logdensity=logp, grad=grad
        ),
    )
    taken = jax.vmap(lambda chain, lane: _begin_doubling(_begin_transition(chain, lane)))(
        chains, taken
    )
    return jax.vmap(_where)(take, taken, lanes), chains


def _refill(logdensity, state, chains, take, job, jobs, data):
    lanes, chains = _assign(logdensity, state.lanes, chains, take, job, jobs, data)
    return state._replace(lanes=lanes), chains


@lru_cache(maxsize=16)
def _compiled(logdensity: Callable, max_depth: int):
    refill = jax.jit(partial(_refill, logdensity), donate_argnums=(0, 1))
    loop = jax.jit(partial(_loop, logdensity, max_depth), donate_argnums=0)
    return refill, loop


def _empty_lanes(num_lanes, dim, dtype, max_depth, data):
    """Lanes without chains."""
    scalar, vector = jnp.zeros((), dtype), jnp.zeros(dim, dtype)
    table = jnp.zeros((max_depth, dim), dtype)
    index = jnp.int32(0)
    key = jax.random.key(0)
    end = Point(vector, vector, vector)
    proposal = Proposal(vector, scalar, vector, scalar, scalar)
    lane = Lane(
        active=False,
        draw=index,
        proposal=proposal,
        tree_key=key,
        energy=scalar,
        depth=index,
        direction=index,
        left=end,
        right=end,
        momentum_sum=vector,
        num_steps=index,
        leaf_keys=key,
        accept_key=key,
        leaf=index,
        sub=proposal,
        sub_momentum_sum=vector,
        checkpoint_momentum=table,
        checkpoint_sum=table,
    )
    chain = Chain(index, key, scalar, vector, jax.tree.map(lambda x: x[0], data))

    return _repeat(lane, num_lanes), _repeat(chain, num_lanes)


def _repeat(tree, n):
    # A separate array for every field, so each can be donated on its own.
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)


def _empty(num_lanes, dim, dtype, max_depth, buffer_size, data):
    lanes, chains = _empty_lanes(num_lanes, dim, dtype, max_depth, data)
    index = jnp.int32(0)
    scalar, vector = jnp.zeros((), dtype), jnp.zeros(dim, dtype)
    record = Record(index, index, vector, scalar, index, False, False, scalar)
    state = State(
        lanes=lanes,
        records=_repeat(record, buffer_size),
        num_records=jnp.zeros((), jnp.int32),
        num_iterations=jnp.zeros((), jnp.int32),
        num_evaluations=jnp.zeros((), jnp.int32),
    )
    return state, chains


def _prepare_queue(logdensity, jobs, data):
    position = np.asarray(jobs.position)
    if not np.issubdtype(position.dtype, np.floating):
        raise TypeError("`jobs.position` must be a floating point array.")
    dtype = position.dtype
    num_jobs = len(position)
    if data is None:
        logdensity = _without_data(logdensity)
        dataset = np.zeros(num_jobs, np.int32)
    elif jobs.dataset is None:
        raise ValueError("`jobs.dataset` must say which slice of `data` each chain reads.")
    else:
        dataset = np.asarray(jobs.dataset, np.int32)
    # The queue stays in host memory; the device only ever sees the chains in the lanes.
    queue = Jobs(
        key=np.asarray(jax.random.key_data(jobs.key)),
        position=position,
        step_size=np.asarray(jobs.step_size, dtype),
        inverse_mass=np.asarray(jobs.inverse_mass, dtype),
        dataset=dataset,
    )
    return logdensity, queue


def _assignment(active, queue, next_job):
    idle = np.flatnonzero(~np.asarray(active))[: len(queue.position) - next_job]
    if not len(idle):
        return None
    take = np.zeros(len(active), bool)
    job = np.zeros(len(active), np.int32)
    take[idle], job[idle] = True, np.arange(next_job, next_job + len(idle))
    return take, job, Jobs(*(x[job] for x in queue)), next_job + len(idle)


def sample(
    logdensity: Callable,
    jobs: Jobs,
    num_draws: int,
    *,
    data=None,
    num_lanes: int | None = None,
    max_depth: int = 10,
    buffer_size: int | None = None,
) -> Draws:
    """Draw `num_draws` samples with every chain in `jobs`.

    Parameters
    ----------
    logdensity
        The log density of one chain's target: `logdensity(position)`, or
        `logdensity(position, data_slice)` when `data` is given.  `position` is a vector.
    jobs
        The chains: keys, initial positions, step sizes and diagonal inverse mass
        matrices, one row per chain.  Transition `t` of a chain is the transition of
        `blackjax.nuts` with the key `jax.random.fold_in(key, t)`.
    num_draws
        Draws per chain.
    data
        Arrays (or a pytree of arrays) with one slice per dataset along their first axis.
        Chain `j` samples `logdensity(position, data[jobs.dataset[j]])`.
    num_lanes
        How many chains run at the same time.  The default is all of them.  With fewer
        lanes than chains, a lane that has finished a chain takes the next one.
    max_depth
        The largest number of doublings of a trajectory (at most `2**max_depth - 1`
        leapfrog steps per transition).
    buffer_size
        How many finished transitions the batched loop may hold before it hands them to
        host memory.  The draws are returned as NumPy arrays and never accumulate on the
        device.  The default is `max(1024, 4 * num_lanes)`.
    """
    logdensity, queue = _prepare_queue(logdensity, jobs, data)
    num_jobs, dim = queue.position.shape
    dtype = queue.position.dtype
    num_lanes = num_jobs if num_lanes is None else min(num_lanes, num_jobs)
    buffer_size = max(1024, 4 * num_lanes) if buffer_size is None else buffer_size
    buffer_size = max(buffer_size, num_lanes)

    shape = (num_jobs, num_draws)
    draws = Draws(
        position=np.empty(shape + (dim,), dtype),
        logdensity=np.empty(shape, dtype),
        num_integration_steps=np.empty(shape, np.int32),
        is_divergent=np.empty(shape, bool),
        is_turning=np.empty(shape, bool),
        acceptance_rate=np.empty(shape, dtype),
        num_iterations=0,
        num_evaluations=0,
    )

    def collect(state):
        """Move the finished transitions from the record buffer to `draws`."""
        records = Record(*(np.asarray(x)[: int(state.num_records)] for x in state.records))
        for name, values in records._asdict().items():
            if name not in ("job", "draw"):
                getattr(draws, name)[records.job, records.draw] = values
        return state._replace(num_records=jnp.zeros((), jnp.int32))

    refill, loop = _compiled(logdensity, max_depth)
    state, chains = _empty(num_lanes, dim, dtype, max_depth, buffer_size, data)
    next_job = num_iterations = num_evaluations = 0
    while True:
        assignment = _assignment(state.lanes.active, queue, next_job)
        if assignment is not None:
            take, job, rows, next_job = assignment
            state, chains = refill(state, chains, take, job, rows, data)
        state = loop(state, chains, jnp.int32(num_draws), next_job < num_jobs)
        num_iterations += int(state.num_iterations)
        num_evaluations += int(state.num_evaluations)
        done = not np.any(state.lanes.active)
        if done or int(state.num_records) + num_lanes > buffer_size:
            state = collect(state)
        if done and next_job == num_jobs:
            break
    return draws._replace(num_iterations=num_iterations, num_evaluations=num_evaluations)


@lru_cache(maxsize=16)
def _without_data(logdensity):
    return lambda position, _: logdensity(position)
