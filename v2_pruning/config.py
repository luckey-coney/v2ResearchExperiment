"""Explicit research parameters; no claimed production thresholds."""
from dataclasses import dataclass, fields
import math
from numbers import Real


@dataclass(frozen=True)
class PruningConfig:
    seed_threshold: float
    grow_threshold: float
    tau: float
    alpha_all: float
    alpha_local: float
    signal_epsilon: float = 1e-8
    denominator_epsilon: float = 1e-8
    norm_epsilon: float = 1e-12
    gain_epsilon: float = 0.0
    coverage_tolerance: float = 1e-6
    gain_chunk_size: int = 64
    query_chunk_size: int = 32
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, values: dict) -> "PruningConfig":
        if not isinstance(values, dict):
            raise ValueError("configuration must be a dictionary")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown, key=str)}")
        try:
            result = cls(**values)
        except TypeError as exc:
            raise ValueError(f"missing or invalid configuration fields: {exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        names = ("seed_threshold", "grow_threshold", "tau", "alpha_all", "alpha_local",
                 "signal_epsilon", "denominator_epsilon", "norm_epsilon",
                 "gain_epsilon", "coverage_tolerance")
        for name in names:
            value = getattr(self, name)
            try:
                valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
            except (OverflowError, TypeError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite real number")
        if not 0 < self.grow_threshold <= self.seed_threshold <= 1:
            raise ValueError("require 0 < grow_threshold <= seed_threshold <= 1")
        if self.tau < 0 or not 0 < self.alpha_all <= 1 or not 0 < self.alpha_local <= 1:
            raise ValueError("require tau >= 0 and 0 < alpha_all, alpha_local <= 1")
        if min(self.signal_epsilon, self.denominator_epsilon, self.norm_epsilon) <= 0:
            raise ValueError("signal, denominator and norm epsilons must be positive")
        tolerance_limit = min(1e-5, min(self.alpha_all, self.alpha_local) * 1e-3)
        if self.gain_epsilon < 0 or not 0 <= self.coverage_tolerance <= tolerance_limit:
            raise ValueError("gain_epsilon >= 0; coverage_tolerance must remain numerical "
                             f"(at most {tolerance_limit:g}), not lower the coverage target")
        for name in ("gain_chunk_size", "query_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_tokens is not None and (isinstance(self.max_tokens, bool)
                or not isinstance(self.max_tokens, int) or self.max_tokens < 0):
            raise ValueError("max_tokens must be None or a nonnegative integer")
