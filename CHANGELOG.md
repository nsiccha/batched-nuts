# Changelog

## 0.1.2

Warmup now supports independent choices of trajectory evidence, diagonal estimator,
initialization, memory, update frequency, and step-size acceptance statistic.
Fixed and adaptive window schedules can count gradients or transitions per chain.

## 0.1.1

Added optional gradient-batched warmup for a diagonal inverse mass and step size.
The result includes ready-to-sample jobs and per-chain costs for each evaluation window.
