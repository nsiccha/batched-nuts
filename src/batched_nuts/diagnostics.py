"""Convergence diagnostics for the draws of many chains.

The sampler returns draws and divergences; whether those draws describe the posterior
is a separate question.  These functions answer it for the arrays that `sample` returns:
`split_rhat` and `ess` per coordinate, following Vehtari, Gelman, Simpson, Carpenter and
Bürkner, "Rank-normalization, folding, and localization: An improved R-hat for assessing
convergence of MCMC" (Bayesian Analysis, 2021), and `summary` for both at once.

The diagnostics are pure NumPy: they read the returned draws, so they work on the output
of `sample`, on a batch collected from `sample_batches`, and on chain arrays from any
other sampler.
"""

from typing import Any, NamedTuple

import numpy as np
from scipy import stats

__all__ = ["Summary", "ess", "rank_normalize", "split_rhat", "summary"]


class Summary(NamedTuple):
    """Diagnostics of one set of draws, one entry per coordinate.

    `rhat` is the rank-normalized split R-hat.  `ess_bulk` and `ess_tail` are the
    effective sample sizes of the draws themselves and of their tails.  `mean` and
    `sd` are those of the draws, over all chains.
    """

    rhat: np.ndarray
    ess_bulk: np.ndarray
    ess_tail: np.ndarray
    mean: np.ndarray
    sd: np.ndarray

    def converged(self, *, max_rhat: float = 1.01, min_ess: float = 100.0) -> bool:
        """Whether every coordinate passes both thresholds."""
        return bool(
            np.all(np.isfinite(self.rhat))
            and np.all(self.rhat <= max_rhat)
            and np.all(self.ess_bulk >= min_ess)
            and np.all(self.ess_tail >= min_ess)
        )


def _as_chains(draws, name="draws", *, min_chains=2, min_draws=4):
    """The draws as a float array of shape (num_chains, num_draws[, dim])."""
    if hasattr(draws, "position"):  # a Draws
        draws = draws.position
    chains = np.asarray(draws, dtype=float)
    if chains.ndim not in (2, 3):
        raise ValueError(
            f"{name} must have shape (num_chains, num_draws) or "
            f"(num_chains, num_draws, dim), got {chains.shape}."
        )
    num_chains, num_draws = chains.shape[:2]
    if num_chains < min_chains:
        raise ValueError(
            f"{name} needs at least {min_chains} chain"
            f"{'' if min_chains == 1 else 's'} to compare, got {num_chains}."
        )
    if num_draws < min_draws:
        raise ValueError(
            f"{name} needs at least {min_draws} draws per chain, got {num_draws}."
        )
    return chains


def _split(chains):
    """Split every chain in half, giving twice as many chains of half the length.

    An odd number of draws drops the last one: the halves then have the same length
    and the split keeps the draws in the order the sampler produced them.
    """
    num_draws = chains.shape[1] // 2
    return np.concatenate([chains[:, :num_draws], chains[:, num_draws : 2 * num_draws]])


def _column(chains):
    """(chains, draws, dim) as (dim, chains, draws), so every coordinate is a plane."""
    return np.moveaxis(chains.reshape(chains.shape[0], chains.shape[1], -1), -1, 0)


def rank_normalize(draws):
    """Replace every draw by the normal score of its rank among all chains.

    This makes the result independent of any monotone transformation of the target and
    keeps a single heavy tail from dominating the variance.  Tied draws get the average
    of the scores of the ranks they share, so the transform is exact for draws that
    repeat.
    """
    chains = _as_chains(draws, min_chains=1, min_draws=1)
    num = chains.shape[0] * chains.shape[1]
    planes = _column(chains)
    scores = np.empty_like(planes)
    reference = stats.norm.ppf((np.arange(1, num + 1) - 0.375) / (num + 0.25))
    for i, plane in enumerate(planes):
        ranks = stats.rankdata(plane.ravel(), method="average")
        scores[i] = np.interp(ranks, np.arange(1, num + 1), reference).reshape(plane.shape)
    return np.moveaxis(scores, 0, -1)


def split_rhat(draws):
    """The rank-normalized split R-hat of every coordinate.

    Each chain is cut in half, the halves are treated as independent chains, and the
    between- and within-chain variances are compared on the rank-normalized draws.  One
    means the chains cannot be told apart.  Values above about 1.01 indicate that the
    chains have not mixed.  A coordinate whose draws do not vary is reported as `nan`.
    """
    chains = _as_chains(draws)
    chains = _split(chains)
    normalized = rank_normalize(chains)
    planes = _column(normalized)
    num_draws = normalized.shape[1]
    within = planes.var(axis=2, ddof=1).mean(axis=1)
    between = num_draws * planes.mean(axis=2).var(axis=1, ddof=1)
    variance = (num_draws - 1) / num_draws * within + between / num_draws
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(variance / within)
    return np.where(within > 0, rhat, np.nan)


def _autocovariance(planes):
    """Biased autocovariance of every chain, (dim, chains, draws)."""
    num_draws = planes.shape[-1]
    size = 1 << int(2 * num_draws - 1).bit_length()
    centered = planes - planes.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=size, axis=-1)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=-1)
    return autocovariance[..., :num_draws] / num_draws


def _effective_size(planes):
    """Effective sample size by Geyer's initial positive sequence, per coordinate."""
    num_chains, num_draws = planes.shape[1:]
    autocovariance = _autocovariance(planes)
    within = autocovariance[..., 0].mean(axis=1) * num_draws / (num_draws - 1)
    between = num_draws * planes.mean(axis=-1).var(axis=1, ddof=1)
    variance = (num_draws - 1) / num_draws * within + between / num_draws
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1 - (within[:, None] - autocovariance.mean(axis=1)) / variance[:, None]
    # Geyer's initial positive sequence: sum the pairs and stop at the first one that
    # is not positive, after making the sequence of pairs decrease monotonically.
    pairs = (
        rho[:, 1 : 1 + 2 * ((num_draws - 1) // 2) : 2]
        + rho[:, 2 : 2 + 2 * ((num_draws - 1) // 2) : 2]
    )
    if pairs.shape[1]:
        pairs = np.minimum.accumulate(pairs, axis=1)
        keep = pairs > 0
        positive = np.where(keep, pairs, 0.0)
        total = 1 + 2 * positive.sum(axis=1)
    else:
        total = np.ones(len(rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        size = num_chains * num_draws / total
    return np.where(variance > 0, size, np.nan)


def ess(draws, *, kind="bulk"):
    """The effective sample size of every coordinate.

    `kind="bulk"` measures the draws and `kind="tail"` the indicator of lying in the
    lower or upper 5% of all draws of that coordinate, taking the smaller of the two.
    A coordinate whose draws do not vary is reported as `nan`.  Chains are cut in half
    as for `split_rhat`, so the size is that of at most twice the draws of one chain.
    """
    if kind not in ("bulk", "tail"):
        raise ValueError("kind must be 'bulk' or 'tail'.")
    chains = _split(_as_chains(draws))
    if kind == "bulk":
        return _effective_size(_column(rank_normalize(chains)))
    plains = _column(chains)
    lower = np.quantile(plains, 0.05, axis=(1, 2))
    upper = np.quantile(plains, 0.95, axis=(1, 2))
    below = (plains <= lower[:, None, None]).astype(float)
    above = (plains >= upper[:, None, None]).astype(float)
    return np.minimum(_effective_size(below), _effective_size(above))


def summary(draws: Any) -> Summary:
    """The diagnostics of `sample`'s draws, or of any array of chains.

    `draws` is a `Draws` or an array of shape (num_chains, num_draws[, dim]).  The
    divergences are not part of the result: they are reported per draw by
    `Draws.is_divergent`.
    """
    chains = _as_chains(draws, name="draws")
    planes = _column(chains)
    return Summary(
        rhat=split_rhat(chains),
        ess_bulk=ess(chains, kind="bulk"),
        ess_tail=ess(chains, kind="tail"),
        mean=planes.mean(axis=(1, 2)),
        sd=planes.std(axis=(1, 2), ddof=1),
    )
