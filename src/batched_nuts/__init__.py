"""NUTS for many chains at once in JAX, batched by gradient evaluation."""

from batched_nuts.metric import MetricFactor
from batched_nuts.nuts import Draws, Jobs, sample
from batched_nuts.policy import (
    AdaptiveWindows,
    DiagonalAdaptation,
    FixedWindows,
    StructuredAdaptation,
)
from batched_nuts.stream import (
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
    "DiagonalAdaptation",
    "FixedWindows",
    "DrawBatch",
    "Draws",
    "Jobs",
    "LoweredStream",
    "MetricFactor",
    "StructuredAdaptation",
    "WarmupEvent",
    "WarmupResult",
    "WindowEvidence",
    "lower_sample_batches",
    "sample",
    "sample_batches",
    "warmup",
]
