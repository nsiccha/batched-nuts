"""Fit and compare metric factors on the same bounded per-chain window."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from batched_nuts.metric import MetricFactor, _apply, _reflect


def _prepare_factor(metric, max_reflections):
    if not isinstance(metric, MetricFactor):
        jobs, dim = metric.shape
        metric = MetricFactor(
            np.ones_like(metric),
            np.empty((jobs, 0, dim), metric.dtype),
            np.empty((jobs, 0, 0), metric.dtype),
            np.sqrt(metric),
        )
    rows = min(max_reflections, metric.scale.shape[1] - 1)
    extra = rows - metric.reflections.shape[1]
    if extra <= 0:
        return metric
    return metric._replace(
        reflections=np.pad(metric.reflections, ((0, 0), (0, extra), (0, 0)))
    )


class _Window(NamedTuple):
    initial: MetricFactor
    points: Any
    count: Any


def _window(metric, capacity):
    dim = metric.scale.shape[0]
    return _Window(
        metric,
        jnp.zeros((capacity, 2 * dim), metric.scale.dtype),
        jnp.int32(0),
    )


def _spreads(x, g, denom):
    return jnp.linalg.norm(x, axis=0) / denom, jnp.linalg.norm(g, axis=0) / denom


def _fit_scale(metric, x, g, denom, score):
    sx, sg = _spreads(
        _apply(metric, x, solve=True), _apply(metric, g, transpose=True), denom
    )
    log_ratio = (jnp.log(sx) - jnp.log(sg)) / 2
    scale = metric.scale * jnp.exp(log_ratio)
    products = sx * sg
    loss = (
        jnp.mean(products) if score == "mean_product" else jnp.sum(jnp.log(products) ** 2)
    )
    valid = jnp.all(jnp.isfinite(scale) & (scale > 0) & (products > 0))
    loss = jnp.where(valid & jnp.isfinite(loss), loss, jnp.inf)
    return metric._replace(scale=scale), loss, jnp.max(log_ratio) - jnp.min(log_ratio)


def _reflection_frame(identity, x, g, denom, limit):
    """Resolve poorly aligned coordinates along gradient singular directions."""
    vectors = identity.reflections
    limit = min(limit, x.shape[1] - 1, x.shape[0] - 1)
    if not limit:
        return identity
    threshold = jnp.log(2.0)

    def bad_columns(x, g):
        sx, sg = _spreads(x, g, denom)
        return jnp.log(sx * sg) ** 2 >= threshold

    def step(state):
        i, vectors, x, g, bad = state
        gradients = jnp.where(bad, g, 0)
        # Thin SVD size is bounded by the window capacity, even at large dimension.
        _, singular, axes = jnp.linalg.svd(gradients, full_matrices=False)
        axes = jnp.where(bad, axes[:10], 0)
        products = (
            jnp.linalg.norm(x @ axes.T, axis=0)
            * jnp.linalg.norm(g @ axes.T, axis=0)
            / denom**2
        )
        usable = (singular[:10] > 0) & (jnp.log(products) ** 2 <= threshold)
        direction = axes[jnp.argmax(usable)]
        pivot = jnp.argmax(jnp.abs(direction))
        v = direction.at[pivot].add(jnp.where(direction[pivot] >= 0, 1.0, -1.0))
        v = jnp.where(jnp.any(usable), v / jnp.linalg.norm(v), 0)
        x, g = _reflect(x, v), _reflect(g, v)
        return i + 1, vectors.at[i].set(v), x, g, bad_columns(x, g) & jnp.any(usable)

    def needed(state):
        return (state[0] < limit) & (jnp.sum(state[4]) >= 2)

    _, vectors, _, _, _ = jax.lax.while_loop(
        needed, step, (jnp.int32(0), vectors, x, g, bad_columns(x, g))
    )
    return identity._replace(reflections=vectors)


def _select(window, current, policy, tolerance):
    n = jnp.minimum(window.count, window.points.shape[0])
    valid = jnp.arange(window.points.shape[0]) < n
    points = jnp.where(valid[:, None], window.points, 0)
    mean = points.sum(axis=0) / jnp.maximum(n, 1)
    x, g = jnp.split(jnp.where(valid[:, None], points - mean, 0), 2, axis=1)
    denom = jnp.sqrt(jnp.maximum(n - 1, 1).astype(x.dtype))
    ones = jnp.ones_like(current.scale)
    identity = MetricFactor(
        ones,
        jnp.zeros_like(current.reflections),
        jnp.eye(current.lower.shape[0], dtype=x.dtype),
        ones,
    )
    rotated = _reflection_frame(identity, x, g, denom, policy.max_reflections)
    best = current
    best_loss = jnp.array(jnp.inf, x.dtype)
    for frame in (identity, window.initial, rotated):
        fitted, loss, _ = _fit_scale(frame, x, g, denom, policy.score)
        better = loss < best_loss
        best = jax.tree.map(lambda a, b, take=better: jnp.where(take, a, b), fitted, best)
        best_loss = jnp.minimum(best_loss, loss)
    _, old_loss, spread = _fit_scale(current, x, g, denom, policy.score)
    if policy.score == "mean_product":
        improvement = jnp.log(old_loss) - jnp.log(best_loss)
    else:
        improvement = jnp.sqrt(old_loss / x.shape[1]) - jnp.sqrt(best_loss / x.shape[1])
    stable = jnp.isfinite(best_loss) & (
        jnp.maximum(spread, improvement) < jnp.log(tolerance)
    )
    return best, stable


def _fit_due(window, metric, due, policy, tolerance):
    """Gather closing windows in small batches; bound the refit's SVD workspace."""
    ready = jax.lax.all_gather(due, "lanes")
    width = ready.size

    def fit():
        if width <= 8:
            return _select(window, metric, policy, tolerance)
        windows, metrics = jax.lax.all_gather((window, metric), "lanes")
        ids = jnp.nonzero(ready, size=width, fill_value=width)[0]
        batch_size = 8

        def fit_batch(i, carry):
            slots = i * batch_size + jnp.arange(batch_size)
            idx = jnp.where(slots < width, ids[jnp.minimum(slots, width - 1)], width)
            subset = jax.tree.map(
                lambda a: a[jnp.minimum(idx, width - 1)], (windows, metrics)
            )
            fits = jax.vmap(lambda w, m: _select(w, m, policy, tolerance))(*subset)
            return jax.tree.map(
                lambda old, new: old.at[idx].set(new, mode="drop"), carry, fits
            )

        batches = (jnp.sum(ready, dtype=jnp.int32) + batch_size - 1) // batch_size
        results = jax.lax.fori_loop(0, batches, fit_batch, (metrics, jnp.zeros_like(ready)))
        lane = jax.lax.axis_index("lanes")
        return jax.tree.map(lambda a: a[lane], results)

    # The predicate is scalar under vmap. Ordinary gradient rounds skip the fit.
    fitted, stable = jax.lax.cond(jnp.any(ready), fit, lambda: (metric, jnp.bool_(False)))
    fitted = jax.tree.map(lambda a, b: jnp.where(due, a, b), fitted, metric)
    return fitted, stable & due


def _update_window(
    window,
    metric,
    policy,
    lane,
    ended,
    boundary,
    fitting,
    tolerance,
    key,
):
    # Uniform reservoir of completed selected proposals. Randomness for retention
    # is separate from the transition keys. Every candidate sees the same rows.
    capacity = window.points.shape[0]
    add = fitting & ended
    count = window.count + add.astype(jnp.int32)
    slot = jnp.where(
        count <= capacity,
        count - 1,
        jax.random.randint(
            jax.random.fold_in(jax.random.fold_in(key, 0x57494E44), lane.draw),
            (),
            0,
            jnp.maximum(count, 1),
        ),
    )
    sample = jnp.concatenate((lane.proposal.position, lane.proposal.grad))
    points = window.points.at[jnp.where(add, slot, capacity)].set(sample, mode="drop")
    window = window._replace(points=points, count=count)
    due = fitting & boundary
    metric, stable = _fit_due(window, metric, due, policy, tolerance)
    return window._replace(count=jnp.where(due, 0, count)), metric, stable
