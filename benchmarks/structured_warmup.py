"""Compare pooled draws from matched diagonal and structured warmup on a Gaussian.

Run with PYTHONPATH=src python benchmarks/structured_warmup.py. Timing starts
with warmup and uses compiled kernels. Every method gets the same number of
gradient rounds, initial positions, and chain keys.
"""

import argparse
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from batched_nuts import (
    DiagonalAdaptation,
    FixedWindows,
    Jobs,
    MetricFactor,
    StructuredAdaptation,
    sample_batches,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=726)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    chains = args.chains
    rho = 0.99
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    precision = jnp.asarray(np.linalg.inv(covariance))

    def target(x):
        return -0.5 * x @ precision @ x

    rng = np.random.default_rng(args.seed)
    starts = rng.multivariate_normal(np.zeros(2), covariance, size=chains)
    jobs = Jobs(
        jax.random.split(jax.random.key(args.seed), chains),
        starts,
        np.full(chains, 0.12),
        np.ones_like(starts),
    )
    # An exact rank-one initial covariance is an oracle for this synthetic target.
    factor = MetricFactor.from_low_rank(
        np.full_like(starts, 1.0 - rho),
        np.ones((chains, 2, 1)),
        np.full((chains, 1, 1), rho),
    )
    supplied = jobs._replace(inverse_mass=factor)
    methods = {
        "diagonal": (
            jobs,
            dict(diagonal=DiagonalAdaptation(evidence="selected", shrinkage=0)),
        ),
        "learned": (jobs, dict(structured=StructuredAdaptation())),
        "supplied-mean": (supplied, dict(structured=StructuredAdaptation())),
        "supplied-log": (
            supplied,
            dict(structured=StructuredAdaptation(score="squared_log")),
        ),
    }
    schedule = FixedWindows((96, 192), 12, final_clock="transitions")

    def run(jobs, options, limit):
        counts = np.zeros(chains, int)
        sums = np.zeros(chains)
        first_time = np.nan
        start = perf_counter()
        for batch in sample_batches(
            target,
            jobs,
            schedule=schedule,
            **options,
            max_depth=6,
            max_rounds=limit,
            rounds_per_batch=256,
        ):
            rec = batch.records
            if len(rec.job) and np.isnan(first_time):
                first_time = perf_counter() - start
            # This projection onto the long axis has known posterior variance one.
            value = rec.position.sum(axis=1) / np.sqrt(2 * (1 + rho))
            counts += np.bincount(rec.job, minlength=chains)
            sums += np.bincount(rec.job, weights=value, minlength=chains)
        elapsed = perf_counter() - start
        total = counts.sum()
        # Variance of the pooled mean using chains as independent clusters.
        # Counts can differ, so use the residual contribution of each chain.
        if total and chains > 1:
            residual = sums - counts * sums.sum() / total
            variance = (chains / (chains - 1)) * (residual @ residual) / total**2
            ess = 1 / variance if variance > 0 else np.nan
        else:
            ess = np.nan
        return first_time, elapsed, total, ess

    print(
        f"device={jax.devices()[0]} chains={chains} rounds={args.rounds} seed={args.seed}"
    )
    print("method          first_s   elapsed_s     draws      ESS      ESS/s  ESS/gradient")
    for name, (start_jobs, options) in methods.items():
        run(start_jobs, options, args.rounds)  # exercise both mixed and frozen kernels
        first, elapsed, draws, ess = run(start_jobs, options, args.rounds)
        print(
            f"{name:14s} {first:8.3f} {elapsed:11.3f} {draws:9d} {ess:8.1f} "
            f"{ess / elapsed:10.1f} {ess / (chains * (args.rounds + 1)):13.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
