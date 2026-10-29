"""Joint capped-coverage greedy selection; K is an output, never a fill target.

Incremental max-cover state and vectorized marginal gains are ported from
third-party jax-coverpruner select.py (MIT), not the paper authors' implementation.
V2 adds region-only local credit, capped H, dynamic stopping and full fallback.
"""
import math
import torch
from torch import Tensor
from .config import PruningConfig
from .coverage import NumericalIssue, as_float_tensor, build_utilities, normalize_relevance
from .regions import build_regions
from .timing import _StageTimer
from .types import CoverageStep, PruningResult, RegionMap, SelectionResult


def _fallback(n, reason, trace=None, order=None):
    # indices describe actual output; order/trace describe the failed attempt.
    initial = [CoverageStep(0, None, 0.0, 0.0, [], 0.0)]
    return SelectionResult(list(range(n)), list(order or []), "fallback_full", reason,
                           trace if trace is not None else initial)


def _region_indices(region_map, n, device):
    if not isinstance(region_map, RegionMap):
        raise NumericalIssue("region_map: expected RegionMap")
    labels = region_map.labels
    if (not isinstance(labels, Tensor) or labels.shape != (n,)
            or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise NumericalIssue("region labels: expected an integer [N] tensor")
    shape = region_map.grid_shape
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape)
            or shape[0] * shape[1] != n):
        raise NumericalIssue("region_map: invalid grid mapping")
    expected = [0] * n
    regions = []
    if not isinstance(region_map.regions, list):
        raise NumericalIssue("region_map: expected a list of regions")
    for label, members in enumerate(region_map.regions, start=1):
        if not isinstance(members, (list, tuple)) or not members:
            raise NumericalIssue("region_map: empty registered region")
        for index in members:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n or expected[index]:
                raise NumericalIssue("region_map: invalid or overlapping region members")
            expected[index] = label
        regions.append(torch.tensor(members, device=device, dtype=torch.long))
    labels = labels.to(device=device, dtype=torch.long)
    if labels.cpu().tolist() != expected:
        raise NumericalIssue("region_map: labels and region members disagree")
    return labels, regions


@torch.no_grad()
def select_tokens(A: Tensor, w: Tensor, region_map: RegionMap, cfg: PruningConfig) -> SelectionResult:
    """Exact all-candidate greedy H. Explicit empty regions means global-only.

    The main prune_visual_tokens entry point instead rejects unexpected B=0.
    CoverageStep(0) is the empty set; fallback never overwrites attempted history.
    """
    cfg.validate()
    n = int(A.shape[0]) if isinstance(A, Tensor) and A.ndim else 0
    order, trace = [], [CoverageStep(0, None, 0.0, 0.0, [], 0.0)]
    try:
        A, w = as_float_tensor(A, "A"), as_float_tensor(w, "w")
        if A.shape != (n, n) or w.shape != (n,):
            raise NumericalIssue("A,w: expected [N,N] and [N]")
        if n == 0:
            return SelectionResult([], [], "no_visual_tokens", None, trace)
        dtype = torch.float64 if torch.float64 in (A.dtype, w.dtype) else torch.float32
        A, w = A.to(dtype=dtype), w.to(device=A.device, dtype=dtype)
        if bool((A < 0).any()) or bool((w < 0).any()):
            raise NumericalIssue("A,w: negative values")
        diag = A.diagonal()
        if bool((A > diag[:, None] + cfg.coverage_tolerance).any()):
            raise NumericalIssue("A: diagonal is not the full-set row maximum")
        labels, regions = _region_indices(region_map, n, A.device)
        trace[0].local_coverages = [0.0] * len(regions)
        weighted_diag = w * diag
        denominators = torch.stack([weighted_diag.sum()] + [weighted_diag[r].sum() for r in regions])
        if (not bool(torch.isfinite(denominators).all())
                or bool((denominators <= cfg.denominator_epsilon).any())):
            raise NumericalIssue("coverage denominator: zero, tiny or nonfinite")
        c_all, c_local = A.new_zeros(n), A.new_zeros(n)
        chosen = torch.zeros(n, dtype=torch.bool, device=A.device)
        targets = A.new_tensor([cfg.alpha_all] + [cfg.alpha_local] * len(regions))
        if not bool(torch.isfinite(targets).all()) or bool((targets <= 0).any()):
            raise NumericalIssue("coverage targets are not positive finite values in the computation dtype")

        def state():
            numerator = torch.stack([(w * c_all).sum()] + [(w[r] * c_local[r]).sum() for r in regions])
            coverages = numerator / denominators
            if (not bool(torch.isfinite(coverages).all())
                    or bool(((coverages < 0) | (coverages > 1 + cfg.coverage_tolerance)).any())):
                raise NumericalIssue("coverage rates: nonfinite or outside [0,1]")
            progress = torch.minimum(coverages / targets, torch.ones_like(targets)).sum()
            if not bool(torch.isfinite(progress)):
                raise NumericalIssue("coverage progress: nonfinite value")
            return coverages, progress

        current, progress = state()
        while True:
            if bool((current + cfg.coverage_tolerance >= targets).all()):
                return SelectionResult(sorted(order), order, "satisfied", None, trace)
            if cfg.max_tokens is not None and len(order) >= cfg.max_tokens:
                return SelectionResult(sorted(order), order, "resource_limited_unsatisfied",
                                       "max_tokens reached before all coverage targets", trace)
            if len(order) == n:
                raise NumericalIssue("all candidates exhausted before coverage targets")
            # A row is a demand, a column a candidate. Each candidate can change
            # at most its own region term, in addition to the global H term.
            local_current = torch.cat((A.new_zeros(1), current[1:]))
            local_denoms = torch.cat((A.new_ones(1), denominators[1:]))
            best_gain, best_index = -math.inf, None
            for start in range(0, n, cfg.gain_chunk_size):
                stop = min(start + cfg.gain_chunk_size, n)
                ids = torch.arange(start, stop, device=A.device)
                candidates = A[:, start:stop]
                delta_all = (w[:, None] * (candidates - c_all[:, None]).clamp_min(0)).sum(dim=0)
                next_all = (current[0] + delta_all / denominators[0]) / cfg.alpha_all
                gains = next_all.clamp_max(1) - (current[0] / cfg.alpha_all).clamp_max(1)
                same_region = (labels[:, None] == labels[ids][None, :]) & (labels[:, None] > 0)
                delta_local = (w[:, None] * (candidates - c_local[:, None]).clamp_min(0)
                               * same_region).sum(dim=0)
                base = local_current[labels[ids]]
                next_local = (base + delta_local / local_denoms[labels[ids]]) / cfg.alpha_local
                gains += next_local.clamp_max(1) - (base / cfg.alpha_local).clamp_max(1)
                if not bool(torch.isfinite(gains).all()):
                    raise NumericalIssue("marginal gains: nonfinite values")
                gains[chosen[ids]] = -torch.inf
                offset = int(torch.argmax(gains))
                gain = float(gains[offset])
                # Ascending chunks and first argmax guarantee minimum-index ties.
                if gain > best_gain:
                    best_gain, best_index = gain, start + offset
            if best_index is None or not math.isfinite(best_gain) or best_gain <= cfg.gain_epsilon:
                raise NumericalIssue("no effective positive marginal gain before targets")
            chosen[best_index] = True
            order.append(best_index)
            c_all = torch.maximum(c_all, A[:, best_index])
            own_region = (labels == labels[best_index]) & (labels > 0)
            c_local = torch.where(own_region, torch.maximum(c_local, A[:, best_index]), c_local)
            current, progress = state()
            trace.append(CoverageStep(len(order), best_index, best_gain, float(current[0]),
                                      current[1:].cpu().tolist(), float(progress)))
    except NumericalIssue as exc:
        return _fallback(n, str(exc), trace, order)


@torch.no_grad()
def prune_visual_tokens(v0: Tensor, raw_relevance: Tensor, grid_shape: tuple[int, int],
                        cfg: PruningConfig, *, timings: dict | None = None) -> PruningResult:
    """Return original indices; optional synchronized stage timings are in ms."""
    cfg.validate()  # Configuration errors propagate instead of masquerading as fallback.
    n = int(v0.shape[0]) if isinstance(v0, Tensor) and v0.ndim else 0
    q = w = regions = None
    try:
        if not isinstance(v0, Tensor) or v0.ndim != 2 or v0.shape[1] == 0:
            raise NumericalIssue("v0: expected [N,D] with D > 0")
        if n == 0:
            return PruningResult(SelectionResult([], [], "no_visual_tokens", None,
                                 [CoverageStep(0, None, 0.0, 0.0, [], 0.0)]), None, None, None)
        if not isinstance(raw_relevance, Tensor) or raw_relevance.shape != (n,):
            raise NumericalIssue("raw_relevance: length does not match v0")
        with _StageTimer(timings, "signal_weights", raw_relevance.device):
            q, w = normalize_relevance(raw_relevance, cfg)
        with _StageTimer(timings, "regions", q.device):
            regions = build_regions(q, grid_shape, cfg)
        if not regions.regions:
            raise NumericalIssue("region generation unexpectedly produced B=0")
        with _StageTimer(timings, "coverage_matrix", v0.device):
            A = build_utilities(v0, cfg)
        with _StageTimer(timings, "selection", A.device):
            selection = select_tokens(A, w, regions, cfg)
    except NumericalIssue as exc:
        selection = _fallback(n, str(exc))
    return PruningResult(selection, regions, q, w)
