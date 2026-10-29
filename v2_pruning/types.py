"""Zero-based indices throughout; trace describes attempted greedy selection."""
from dataclasses import dataclass
from torch import Tensor


@dataclass
class RegionMap:
    labels: Tensor  # [N], 0 = background, 1..B = registered regions
    regions: list[list[int]]
    seeds: list[int]
    grid_shape: tuple[int, int]


@dataclass
class CoverageStep:
    step: int
    selected_index: int | None
    gain: float
    global_coverage: float
    local_coverages: list[float]
    progress: float


@dataclass
class SelectionResult:
    indices: list[int]  # Final retained set, restored to original order.
    order: list[int]  # Attempted greedy picks; fallback retains all regardless.
    status: str
    reason: str | None
    trace: list[CoverageStep]  # Never reinterpreted as achieved fallback coverage.


@dataclass
class PruningResult:
    selection: SelectionResult
    regions: RegionMap | None
    q: Tensor | None
    w: Tensor | None
