"""Gradient evaluations and wall time: `batched_nuts.sample` and vmapped `blackjax.nuts`.

The task is a small simulation study: logistic regressions on simulated datasets, a few
chains per dataset.  The predictors of a dataset are correlated, more strongly in some
datasets than in others, so some posteriors need trajectories of 7 leapfrog steps and
others of more than 100.  Every chain is first adapted with `blackjax.window_adaptation`
(step size and diagonal mass matrix), one chain after the other on the CPU; warm-up is
not timed.  Then both samplers draw the same samples: they make the same transitions
(see the tests), so they do the same useful work, the leapfrog steps of the trajectories.
What differs is what they pay for it.

* vmapped `blackjax.nuts` runs the chains of a batch in step: every transition lasts as
  long as the longest trajectory in the batch.  It pays at least
  `batch size * sum over transitions of the longest trajectory` gradient evaluations.
  (At least: within a transition, every doubling also lasts as long as its longest
  subtree.  The number reported here is this lower bound.)
* `batched_nuts.sample` pays `number of lanes * iterations of its loop`.

Each sampler is compiled on a short run first, so compilation is not timed.  On a CPU
both run on one thread unless XLA_FLAGS says otherwise.

    python benchmarks/benchmark.py [--datasets 64] [--chains 4] [--lanes 64] ...
"""

import argparse
import json
import os
import time

os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import blackjax  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import batched_nuts  # noqa: E402

from blackjax_baseline import make_blackjax_sampler  # noqa: E402


def logdensity(coefficients, data):
    """Logistic regression with independent normal(0, 2) priors."""
    logits = data["x"] @ coefficients
    loglikelihood = jnp.sum(data["y"] * logits - jnp.logaddexp(0.0, logits))
    return loglikelihood - 0.5 * jnp.sum((coefficients / 2) ** 2)


def simulate(key, num_datasets, num_rows, dim):
    """Datasets whose predictors have a pairwise correlation between 0 and 0.99."""
    correlation_key, common_key, own_key, coefficient_key, y_key = jax.random.split(key, 5)
    correlation = 1 - 10 ** jax.random.uniform(
        correlation_key, (num_datasets, 1, 1), minval=-2.0, maxval=0.0
    )
    common = jax.random.normal(common_key, (num_datasets, num_rows, 1))
    own = jax.random.normal(own_key, (num_datasets, num_rows, dim))
    x = jnp.sqrt(correlation) * common + jnp.sqrt(1 - correlation) * own
    coefficients = jax.random.normal(coefficient_key, (num_datasets, dim))
    logits = jnp.einsum("knd,kd->kn", x, coefficients)
    y = jax.random.bernoulli(y_key, jax.nn.sigmoid(logits)).astype(x.dtype)
    return dict(x=x, y=y)


def adapt(key, data, dataset, num_steps, dim):
    """Warm up every chain with blackjax: its position, step size and inverse mass."""

    def one_chain(arguments):
        key, dataset = arguments
        data_slice = jax.tree.map(lambda x: x[dataset], data)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, lambda position: logdensity(position, data_slice)
        )
        start_key, warmup_key = jax.random.split(key)
        start = 0.1 * jax.random.normal(start_key, (dim,))
        (state, parameters), _ = warmup.run(warmup_key, start, num_steps=num_steps)
        return state.position, parameters["step_size"], parameters["inverse_mass_matrix"]

    keys = jax.random.split(key, len(dataset))
    return jax.jit(lambda *xs: jax.lax.map(one_chain, xs))(keys, dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", type=int, default=64)
    parser.add_argument("--chains", type=int, default=4, help="chains per dataset")
    parser.add_argument("--lanes", type=int, default=64, help="chains run at a time")
    parser.add_argument("--rows", type=int, default=500, help="rows of a dataset")
    parser.add_argument("--dim", type=int, default=10, help="predictors")
    parser.add_argument("--draws", type=int, default=200, help="draws per chain")
    parser.add_argument("--warmup", type=int, default=300, help="warm-up steps per chain")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the results to this JSON file")
    args = parser.parse_args()
    num_chains = args.datasets * args.chains
    if num_chains % args.lanes:
        parser.error("--lanes must divide the number of chains")

    data_key, warmup_key, sampling_key = jax.random.split(jax.random.key(args.seed), 3)
    data = simulate(data_key, args.datasets, args.rows, args.dim)
    dataset = jnp.repeat(jnp.arange(args.datasets), args.chains)
    start = time.perf_counter()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        on_cpu = jax.device_put((warmup_key, data, dataset), cpu)
        adapted = jax.block_until_ready(adapt(*on_cpu, args.warmup, args.dim))
    position, step_size, inverse_mass = jax.device_put(adapted, jax.devices()[0])
    print(f"warm-up: {time.perf_counter() - start:.0f} s", flush=True)
    jobs = batched_nuts.Jobs(
        key=jax.random.split(sampling_key, num_chains),
        position=position,
        step_size=step_size,
        inverse_mass=inverse_mass,
        dataset=dataset,
    )
    batches = [
        jax.tree.map(lambda x, i=i: x[i : i + args.lanes], jobs)
        for i in range(0, num_chains, args.lanes)
    ]

    def run_blackjax(batches, num_draws):
        return [blackjax_sample(batch, num_draws, data) for batch in batches]

    def run_batched(groups, num_draws):
        return [
            batched_nuts.sample(
                logdensity,
                group,
                num_draws,
                data=data,
                num_lanes=args.lanes,
                max_depth=args.max_depth,
            )
            for group in groups
        ]

    def timed(run, groups):
        start = time.perf_counter()
        result = jax.block_until_ready(run(groups, args.draws))
        return result, time.perf_counter() - start

    blackjax_sample = make_blackjax_sampler(logdensity, args.max_depth)
    jax.block_until_ready(run_blackjax(batches[:1], args.draws))  # compile
    run_batched(batches[:1], 2)  # compile

    reference, seconds = timed(run_blackjax, batches)
    reference = {
        name: np.concatenate([np.asarray(r[name]) for r in reference])
        for name in reference[0]
    }
    steps = reference["num_integration_steps"]
    in_batches = steps.reshape(len(batches), args.lanes, args.draws)
    results = [
        dict(
            sampler="vmapped blackjax.nuts, in batches",
            paid=int(args.lanes * in_batches.max(axis=1).sum()),
            useful=int(steps.sum()),
            seconds=seconds,
        )
    ]
    for label, groups in [("in batches", batches), ("with a queue", [jobs])]:
        draws, seconds = timed(run_batched, groups)
        position = np.concatenate([d.position for d in draws])
        results.append(
            dict(
                sampler=f"batched_nuts, {label}",
                paid=int(args.lanes * sum(d.num_iterations for d in draws)),
                useful=int(sum(d.num_evaluations for d in draws)),
                seconds=seconds,
                same_trajectories=bool(
                    np.array_equal(
                        np.concatenate([d.num_integration_steps for d in draws]), steps
                    )
                ),
                draws_equal_bit_for_bit=float(
                    np.mean(np.all(position == reference["position"], axis=-1))
                ),
                largest_difference=float(np.max(np.abs(position - reference["position"]))),
            )
        )

    per_chain = steps.mean(axis=1)
    result = dict(
        device=str(jax.devices()[0]),
        jax=jax.__version__,
        blackjax=blackjax.__version__,
        **vars(args),
        mean_trajectory=float(steps.mean()),
        shortest_and_longest_chain_mean=[float(per_chain.min()), float(per_chain.max())],
        divergent=float(reference["is_divergent"].mean()),
        results=results,
    )
    print(json.dumps(result, indent=1))
    print(f"\n{'':36}{'paid':>12}{'useful':>8}{'seconds':>9}{'ms per batched step':>21}")
    for row in results:
        per_step = 1e3 * row["seconds"] * args.lanes / row["paid"]
        share = row["useful"] / row["paid"]
        print(
            f"{row['sampler']:36}{row['paid']:>12,}{share:>8.0%}"
            f"{row['seconds']:>9.1f}{per_step:>21.3f}"
        )
    if args.out:
        with open(args.out, "w") as file:
            json.dump(result, file, indent=1)


if __name__ == "__main__":
    main()
