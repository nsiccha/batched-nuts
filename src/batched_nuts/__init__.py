"""NUTS for many chains at once in JAX, batched by gradient evaluation."""

from batched_nuts.branching import Branch, Branching, BranchOrigin, Parent
from batched_nuts.grouped import GroupedDensity, GroupFactor
from batched_nuts.hierarchical import HierarchicalStart, hierarchical_start
from batched_nuts.metric import MetricFactor
from batched_nuts.nuts import Draws, Jobs, sample
from batched_nuts.policy import (
    AdaptiveWindows,
    DiagonalAdaptation,
    FixedWindows,
    StructuredAdaptation,
)
from batched_nuts.stream import (
    BranchBatch,
    DrawBatch,
    LoweredStream,
    WarmupEvent,
    WindowEvidence,
    lower_sample_batches,
    sample_batches,
)
from batched_nuts.warmup import WarmupResult, warmup

__all__ = [
    "AdaptiveWindows",
    "Branch",
    "BranchBatch",
    "Branching",
    "BranchOrigin",
    "Parent",
    "DiagonalAdaptation",
    "FixedWindows",
    "DrawBatch",
    "Draws",
    "GroupFactor",
    "GroupedDensity",
    "HierarchicalStart",
    "Jobs",
    "LoweredStream",
    "MetricFactor",
    "StructuredAdaptation",
    "WarmupEvent",
    "WarmupResult",
    "WindowEvidence",
    "hierarchical_start",
    "lower_sample_batches",
    "sample",
    "sample_batches",
    "warmup",
]
