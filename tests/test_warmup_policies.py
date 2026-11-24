import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_nuts import AdaptiveWindows, DiagonalAdaptation, FixedWindows, Jobs, warmup
from batched_nuts.nuts import _advance, _assign, _begin_next, _empty_lanes

from targets import gaussian_jobs


def density(q):
    return -0.5 * (q[0] ** 2 + (q[1] - 0.5 * q[0] ** 2) ** 2)


def original_tree(jobs):
    """Tabulate one trajectory under the public sampler's initial metric."""
    key = jax.random.key_data(jax.vmap(jax.random.split)(jobs.key)[:, 0])
    rows = jobs._replace(key=key, dataset=np.zeros(1, np.int32))
    fn = lambda q, _: density(q)
    lane, chain = _empty_lanes(1, 2, jobs.position.dtype, 4, None)
    lane, chain = _assign(fn, lane, chain, jnp.array([True]), jnp.array([0]), rows, None)
    lane, chain = jax.tree.map(lambda x: x[0], (lane, chain))
    points = [np.r_[lane.proposal.position, lane.proposal.grad]]
    accepted = [0]
    weights = np.ones(1)
    log_weights, subtree = [], []

    @jax.jit
    def advance(lane):
        updated, record, ended, finished, point, (_, prob) = _advance(fn, 4, chain, lane)
        logw = (
            lane.energy
            + density(point.position)
            - jnp.dot(point.momentum, point.momentum) / 2
        )
        return (
            _begin_next(chain, updated, ended, finished),
            record,
            ended,
            finished,
            point,
            logw,
            prob,
        )

    logs = []
    for i in range(1, 16):
        lane, record, ended, finished, point, logw, prob = advance(lane)
        points.append(np.r_[point.position, point.grad])
        logs.append(float(logw))
        subtree.append(i)
        log_weights.append(float(logw))
        if finished:
            relative = np.exp(log_weights - np.max(log_weights))
            weights = np.r_[weights * (1 - prob), prob * relative / sum(relative)]
            accepted.extend(subtree)
            log_weights, subtree = [], []
        if ended:
            break
    assert ended and any(np.asarray(logs) > -np.log(100.0))
    # The exact probability in the main tree differs from a softmax over all leaves.
    return np.array(points), weights, accepted, np.array(logs), np.asarray(record.position)


@pytest.mark.parametrize(
    "source", ["nuts", "selected", "eligible_selected", "eligible_leaves"]
)
@pytest.mark.parametrize("estimator", ["position", "position_score"])
def test_public_fit_against_explicit_tree(source, estimator):
    jobs = Jobs(
        jax.random.split(jax.random.key(32), 1),
        np.array([[0.3, 0.5]]),
        np.array([0.32]),
        np.ones((1, 2)),
    )
    points, weights, accepted, logs, selected = original_tree(jobs)
    if source == "nuts":
        data = np.vstack((points[0], points[accepted]))
        w = np.r_[1.0, weights] / 2
    elif source == "eligible_leaves":
        data = np.vstack((points[0], points[1:][logs > -np.log(100.0)]))
        w = np.ones(len(data)) / len(data)
    else:
        chosen = points[np.all(np.isclose(points[:, :2], selected), axis=1)][0]
        data = np.array([points[0], chosen])
        w = np.array([0.5, 0.5])
    variance = w @ (data - w @ data) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (
            variance[:2]
            if estimator == "position"
            else np.sqrt(variance[:2] / variance[2:])
        )
    m = np.where(np.isfinite(m) & (m > 0), m, 1)
    policy = DiagonalAdaptation(
        evidence=source, estimator=estimator, include_initial=True, shrinkage=0
    )
    fit = warmup(
        density,
        jobs,
        diagonal=policy,
        schedule=FixedWindows((1,), 1, clock="transitions"),
        max_depth=4,
    )
    np.testing.assert_allclose(fit.jobs.inverse_mass[0], m, atol=1e-12)
    np.testing.assert_array_equal(fit.num_transitions, [[1, 1]])


def test_stability_refill_independent_monitor_and_final_clock():
    fn, jobs, data = gaussian_jobs(dim=4)
    jobs = jobs._replace(inverse_mass=jnp.ones_like(jobs.position))
    diag = DiagonalAdaptation(
        evidence="eligible_selected",
        variance="pre_update",
        initial="gradient",
        include_initial=True,
        shrinkage=0,
        memory="overlapping",
        switch_interval=7,
        switch_clock="transitions",
        update_interval=1,
    )
    schedule = AdaptiveWindows(64, max_windows=4, final_window=8, evidence="nuts")
    kwargs = dict(
        data=data,
        max_depth=4,
        diagonal=diag,
        schedule=schedule,
        acceptance="late_symmetric",
        restart_dual=False,
    )
    all_lanes = warmup(fn, jobs, **kwargs)
    queue = warmup(fn, jobs, num_lanes=2, **kwargs)
    for field in queue._fields[1:]:
        np.testing.assert_array_equal(getattr(queue, field), getattr(all_lanes, field))
    np.testing.assert_allclose(queue.jobs.inverse_mass, all_lanes.jobs.inverse_mass)
    np.testing.assert_allclose(queue.jobs.step_size, all_lanes.jobs.step_size)
    np.testing.assert_allclose(
        queue.jobs.inverse_mass, np.asarray(data["scale"][jobs.dataset]) ** 2
    )
    assert np.all(queue.num_evaluations[:, :2] >= [64, 128])
    assert np.all(queue.num_evaluations[:, 2:-1] == 0)
    assert np.all(queue.num_transitions[:, -1] == 8)


def test_unidentified_windows_do_not_report_a_successful_stop():
    jobs = Jobs(
        jax.random.split(jax.random.key(23), 1),
        np.array([[1.0, 2.0]]),
        np.array([0.1]),
        np.ones((1, 2)),
    )
    schedule = AdaptiveWindows(
        1, growth=1, max_windows=2, clock="transitions", final_window=1
    )
    with pytest.raises(RuntimeError, match="max_windows"):
        warmup(
            density,
            jobs,
            diagonal=DiagonalAdaptation(evidence="selected"),
            schedule=schedule,
            max_depth=2,
        )


def test_fixed_metric_uses_supplied_scale_and_calibrates_step():
    fn, jobs, data = gaussian_jobs(dim=3)
    fit = warmup(
        fn,
        jobs,
        data=data,
        diagonal=DiagonalAdaptation(estimator="fixed"),
        schedule=FixedWindows((10,), 4, clock="transitions"),
        max_depth=4,
        acceptance="symmetric",
        restart_dual=False,
    )
    np.testing.assert_array_equal(fit.jobs.inverse_mass, jobs.inverse_mass)
    assert np.all(fit.num_transitions == [10, 4])
    assert np.all(fit.jobs.step_size > 0)


def test_bad_combinations_and_budget_overflow():
    fn, jobs, data = gaussian_jobs(dim=3)
    with pytest.raises(ValueError, match="either"):
        warmup(fn, jobs, data=data, schedule=FixedWindows(), window_evaluations=(5, 2))
    with pytest.raises(ValueError, match="learned diagonal"):
        warmup(
            fn,
            jobs,
            data=data,
            diagonal=DiagonalAdaptation(estimator="fixed"),
            schedule=AdaptiveWindows(),
        )
    with pytest.raises(ValueError, match="int32"):
        warmup(fn, jobs, data=data, schedule=FixedWindows((2**30,), 1, clock="transitions"))


def test_overlapping_pre_update_fits_on_its_own_clock():
    """Recompute every installed diagonal from recorded draws, including promotions."""
    from batched_nuts.warmup import _initial, _warm_step

    jobs = Jobs(
        jax.random.key_data(jax.random.split(jax.random.key(17), 1)),
        np.array([[0.5, -0.25]]),
        np.array([0.16]),
        np.ones((1, 2)),
        np.zeros(1, int),
    )
    fn = lambda q, _: density(q)
    lanes, chains = _empty_lanes(1, 2, jobs.position.dtype, 4, None)
    lanes, chains = _assign(
        fn, lanes, chains, jnp.array([True]), jnp.array([0]), jobs, None
    )
    lane, chain = jax.tree.map(lambda x: x[0], (lanes, chains))
    stats = _initial(
        lane.proposal.position,
        lane.proposal.grad,
        chain.step_size,
        chain.inverse_mass,
        2,
        True,
    )
    diagonal = DiagonalAdaptation(
        evidence="selected",
        variance="pre_update",
        include_initial=True,
        memory="overlapping",
        shrinkage=0,
        switch_interval=4,
        switch_clock="transitions",
        update_interval=6,
        update_clock="gradients",
    )
    schedule = FixedWindows((15,), 3, clock="transitions")

    @jax.jit
    def run_one(lane, stats):
        draw = lane.draw
        return jax.lax.while_loop(
            lambda s: s[0].draw == draw,
            lambda s: _warm_step(
                fn,
                4,
                diagonal,
                schedule,
                "late_symmetric",
                False,
                jnp.array([15, 3]),
                jnp.array(0.8),
                chain,
                *s,
            ),
            (lane, stats),
        )

    points = [np.r_[np.array(lane.proposal.position), np.array(lane.proposal.grad)]]
    background = list(points)
    metric = np.ones(2)
    clock = 0
    for t in range(1, 19):
        before = int(stats.evaluations.sum())
        lane, stats = run_one(lane, stats)
        point = np.r_[np.array(lane.proposal.position), np.array(lane.proposal.grad)]
        if t <= 15:
            points.append(point)
            background.append(point)
            if t % 4 == 0:
                points, background = background, []
            clock += int(stats.evaluations.sum()) - before
            if t == 15 or clock >= 6:
                var = sum(
                    (
                        (v - np.mean(points[:i], axis=0)) ** 2
                        for i, v in enumerate(points)
                        if i
                    ),
                    start=np.zeros(4),
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    fit = np.sqrt(var[:2] / var[2:])
                metric = np.where(np.isfinite(fit) & (fit > 0), fit, metric)
                clock = 0
        np.testing.assert_allclose(stats.metric, metric, rtol=1e-10)
    assert not bool(lane.active)
