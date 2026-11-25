"""Independent choices for fitting a diagonal metric and scheduling warmup."""

from dataclasses import dataclass
from math import isfinite
from operator import index

SOURCES = ("selected", "eligible_selected", "nuts", "eligible_leaves")
CLOCKS = ("gradients", "transitions")


def _choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}.")


def _positive(value, name):
    if isinstance(value, bool) or index(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return index(value)


@dataclass(frozen=True)
class DiagonalAdaptation:
    """Select metric evidence, estimator, initialization, memory and installation."""

    evidence: str = "nuts"
    estimator: str = "position_score"
    initial: str = "provided"
    memory: str = "window"
    update_interval: int | str = "window"
    switch_interval: int | str = "window"
    shrinkage: float = 3.0
    variance: str = "population"
    include_initial: bool = False
    update_clock: str = "transitions"
    switch_clock: str | None = None

    def __post_init__(self):
        _choice("evidence", self.evidence, SOURCES)
        _choice("variance", self.variance, ("population", "pre_update"))
        _validate_clocks(self.update_clock, self.switch_clock)
        _choice("estimator", self.estimator, ("position", "position_score", "fixed"))
        _choice("initial", self.initial, ("provided", "gradient"))
        _choice("memory", self.memory, ("window", "overlapping", "cumulative"))
        for name in ("update_interval", "switch_interval"):
            value = getattr(self, name)
            if value != "window":
                _positive(value, name)
        if self.memory == "cumulative" and self.switch_interval != "window":
            raise ValueError("A cumulative estimate keeps its history across switches.")
        if not isfinite(self.shrinkage) or self.shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative.")


@dataclass(frozen=True)
class FixedWindows:
    """Fit in each metric window and then tune only the step size in the final window."""

    metric_windows: tuple[int, ...] = (1000, 2000)
    final_window: int = 1000
    clock: str = "gradients"
    final_clock: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "metric_windows", tuple(self.metric_windows))
        self.lengths()

    def lengths(self):
        _validate_clocks(self.clock, self.final_clock)
        if not self.metric_windows:
            raise ValueError("Supply at least one metric window.")
        return tuple(
            _positive(v, "window budgets")
            for v in (*self.metric_windows, self.final_window)
        )


@dataclass(frozen=True)
class AdaptiveWindows:
    """End metric fitting when consecutive disjoint windows agree on diagonal shape."""

    first_window: int = 1000
    growth: int = 2
    tolerance: float = 2.0
    max_windows: int = 10
    final_window: int = 45
    clock: str = "gradients"
    final_clock: str | None = "transitions"
    evidence: str | None = None

    def __post_init__(self):
        _validate_clocks(self.clock, self.final_clock)
        _choice("evidence", self.evidence, (None, *SOURCES))
        _positive(self.first_window, "first_window")
        _positive(self.final_window, "final_window")
        _positive(self.growth, "growth")
        if index(self.max_windows) < 2:
            raise ValueError("max_windows must allow at least two disjoint metric windows.")
        if not isfinite(self.tolerance) or self.tolerance <= 1:
            raise ValueError("tolerance must be finite and greater than one.")

    def lengths(self):
        return tuple(
            index(self.first_window) * index(self.growth) ** i
            for i in range(self.max_windows)
        ) + (index(self.final_window),)


def _validate_clocks(clock, final_clock):
    _choice("clock", clock, ("gradients", "transitions"))
    _choice("final_clock", final_clock, (None, "gradients", "transitions"))


@dataclass(frozen=True)
class StructuredAdaptation:
    """Fit three metric frames to a bounded sample of completed NUTS transitions."""

    score: str = "mean_product"
    capacity: int = 64
    max_reflections: int = 8

    def __post_init__(self):
        _choice("score", self.score, ("mean_product", "squared_log"))
        if _positive(self.capacity, "capacity") < 2:
            raise ValueError("capacity must allow at least two observations.")
        if isinstance(self.max_reflections, bool) or index(self.max_reflections) < 0:
            raise ValueError("max_reflections must be a nonnegative integer.")
