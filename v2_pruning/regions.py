"""V2 four-neighbour seed/threshold components, with no region filtering.

RoRA's _rora_aar_mask in patch.py motivates max-pool grid operations only.
Its fixed square-radius Top-K AAR rule is NOT used here. Pooling below is a
four-neighbour cross; equal connected plateaus receive one smallest-index seed.
"""
from collections import deque
import torch
import torch.nn.functional as F
from torch import Tensor
from .config import PruningConfig
from .coverage import NumericalIssue, as_float_tensor
from .types import RegionMap


def build_regions(q: Tensor, grid_shape: tuple[int, int], cfg: PruningConfig) -> RegionMap:
    cfg.validate()
    q = as_float_tensor(q, "q")
    if q.ndim != 1 or bool(((q < 0) | (q > 1)).any()):
        raise NumericalIssue("q: expected [N] within [0,1]")
    if (not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in grid_shape)
            or grid_shape[0] * grid_shape[1] != q.numel()):
        raise NumericalIssue("grid_shape: no valid two-dimensional spatial mapping")
    height, width = grid_shape
    grid = q.reshape(1, 1, height, width)
    neighbour_max = torch.maximum(F.max_pool2d(grid, (3, 1), 1, (1, 0)),
                                  F.max_pool2d(grid, (1, 3), 1, (0, 1)))
    peaks = (grid >= neighbour_max).flatten().cpu().tolist()
    values = q.cpu().tolist()

    def neighbours(index):
        row, col = divmod(index, width)
        return [r * width + c for r, c in ((row - 1, col), (row, col - 1),
                (row, col + 1), (row + 1, col)) if 0 <= r < height and 0 <= c < width]

    def component(start, eligible, seen):
        queue, members = deque([start]), []
        seen.add(start)
        while queue:
            index = queue.popleft()
            members.append(index)
            for other in neighbours(index):
                if other not in seen and eligible(other):
                    seen.add(other)
                    queue.append(other)
        return sorted(members)

    seen, seeds = set(), []
    for index, value in enumerate(values):
        if index not in seen and value >= cfg.seed_threshold:
            plateau = component(index, lambda other: values[other] == value, seen)
            if all(peaks[member] for member in plateau):
                seeds.append(plateau[0])
    seen, regions = set(), []
    seed_set = set(seeds)
    for index, value in enumerate(values):
        if index not in seen and value >= cfg.grow_threshold:
            members = component(index, lambda other: values[other] >= cfg.grow_threshold, seen)
            if seed_set.intersection(members):
                regions.append(members)
    labels = torch.zeros(q.numel(), dtype=torch.long, device=q.device)
    for label, members in enumerate(regions, start=1):
        labels[members] = label
    return RegionMap(labels, regions, seeds, (height, width))
