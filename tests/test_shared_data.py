"""One runtime dataset can serve every lane, including refills and branches."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_nuts import (
    Branch,
    Branching,
    FixedWindows,
    Jobs,
    lower_sample_batches,
    sample,
    sample_batches,
    warmup,
)


def jobs(width):
    positions = np.zeros((width, 1))
    return Jobs(
        jax.random.split(jax.random.key(17), width),
        positions,
        np.full(width, 0.25),
        np.ones_like(positions),
    )


def density(q, data):
    return -0.5 * jnp.sum((q - data["observations"]) ** 2) / data["scale"] ** 2


def inputs():
    raw = {"observations": np.array([1.0, -0.25]), "scale": np.float64(1.5)}
    tables = jax.tree.map(lambda x: np.stack([x + 10, x, x]), raw)
    queue = jobs(3)
    # Rows 1 and 2 hold identical values, so the mixed path is a numerical control.
    return (
        (queue._replace(dataset=np.array([1, 2, 1])), tables),
        (queue._replace(dataset=np.full(3, 1)), tables),
        (queue, raw),
    )


def test_shared_warmup_and_sampling_with_lane_refills():
    results = []
    for queue, data in inputs():
        options = dict(num_lanes=2, max_depth=2, data=data)
        trained = warmup(density, queue, schedule=FixedWindows((3,), 2), **options)
        np.testing.assert_array_equal(trained.jobs.dataset, queue.dataset)
        draws = sample(density, trained.jobs, 8, **options)
        results.append(
            (
                trained.jobs.position,
                trained.jobs.step_size,
                trained.jobs.inverse_mass,
                *trained[1:],
                *draws,
            )
        )
    for result in results[1:]:
        for actual, expected in zip(result, results[0], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("schedule", [None, FixedWindows((3,), 2)])
def test_branched_chains_read_the_shared_dataset(schedule):
    def choose(batch, parents, room):
        parent = next((p for p in parents if p.job == 0 and p.draw_round is not None), None)
        return [] if parent is None else [Branch(parent.job, parent.draw_round)]

    policy = Branching(jax.random.key(23), 1, 4, choose)
    controls = []
    for queue, data in inputs():
        batches = list(
            sample_batches(
                density,
                queue,
                data=data,
                branching=policy,
                schedule=schedule,
                max_depth=2,
                buffer_size=4,
                max_rounds=40,
            )
        )
        (origin,) = [origin for batch in batches for origin in batch.branches]
        assert origin.dataset == (0 if queue.dataset is None else 1)
        records = jax.tree.map(
            lambda *xs: np.concatenate(xs), *(b.records for b in batches)
        )
        assert np.any(records.job == origin.job)
        controls.append(
            (
                origin.parent,
                origin.admission_round,
                np.concatenate([b.completion_rounds for b in batches]),
                *records,
            )
        )
    for result in controls[1:]:
        for actual, expected in zip(result, controls[0], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("width", [2, 5])
@pytest.mark.parametrize("schedule", [None, FixedWindows((2,), 2)])
def test_startup_and_round_take_one_runtime_observation_array(width, schedule):
    options = dict(max_depth=2, schedule=schedule)
    queue = jobs(width)
    raw = {"observations": np.arange(128, dtype=np.float64), "scale": np.float64(1)}
    tables = jax.tree.map(lambda x: np.stack([x + 1, x]), raw)
    direct = lower_sample_batches(density, queue, data=raw, **options)
    indexed = lower_sample_batches(
        density, queue._replace(dataset=np.ones(width, np.int32)), data=tables, **options
    )
    changed = lower_sample_batches(
        density, queue, data=jax.tree.map(lambda x: x + 2, raw), **options
    )
    for stage, same_row, new_values in zip(direct, indexed, changed, strict=True):
        assert stage.as_text() == same_row.as_text() == new_values.as_text()
        input_shapes = [a.shape for a in jax.tree.leaves(stage.in_avals)]
        output_shapes = [a.shape for a in jax.tree.leaves(stage.out_info)]
        assert input_shapes.count((128,)) == 1
        assert (width, 128) not in input_shapes + output_shapes
