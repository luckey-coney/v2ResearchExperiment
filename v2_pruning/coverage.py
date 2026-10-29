"""PyTorch port of third-party jax-coverpruner coverage.py (MIT).

Source: teddytennant/jax-coverpruner, e5ae93963fb164dd9bcd45de688eaba97be3d5d5.
This is not code released by the CoverPruner paper's authors. V2 adds strict
numerical guards and dynamic joint coverage; see THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import Tensor
from .config import PruningConfig


class NumericalIssue(ValueError):
    """Invalid sample data; callers must retain the complete visual sequence."""


def as_float_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.is_complex() or value.dtype == torch.bool:
        raise NumericalIssue(f"{name}: expected a real numeric tensor")
    result = value.detach().to(dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise NumericalIssue(f"{name}: nonfinite values")
    return result


def normalize_relevance(raw: Tensor, cfg: PruningConfig) -> tuple[Tensor, Tensor]:
    cfg.validate()
    raw = as_float_tensor(raw, "raw_relevance")
    if raw.ndim != 1 or raw.numel() == 0:
        raise NumericalIssue("raw_relevance: expected a nonempty [N] tensor")
    span = raw.max() - raw.min()
    if not bool(torch.isfinite(span)) or float(span) <= cfg.signal_epsilon:
        raise NumericalIssue("raw_relevance: degenerate dynamic range")
    q = (raw - raw.min()) / span
    w = torch.exp(cfg.tau * q)
    if not bool(torch.isfinite(q).all() & torch.isfinite(w).all()):
        raise NumericalIssue("normalized relevance or demand weights are nonfinite")
    return q, w


def build_utilities(v0: Tensor, cfg: PruningConfig) -> Tensor:
    cfg.validate()
    v0 = as_float_tensor(v0, "v0")
    if v0.ndim != 2 or v0.shape[1] == 0:
        raise NumericalIssue("v0: expected [N,D] with D > 0")
    if v0.shape[0] == 0:
        return v0.new_empty((0, 0))
    norms = torch.linalg.vector_norm(v0, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= cfg.norm_epsilon).any()):
        raise NumericalIssue("v0: zero, tiny or nonfinite vector norm")
    unit = v0 / norms
    # Exact identical directions mathematically give zero centered utility.
    # Avoid manufacturing positive diagonal utility from float32 dot roundoff.
    if torch.equal(unit, unit[:1].expand_as(unit)):
        return v0.new_zeros((v0.shape[0], v0.shape[0]))
    # Demand i is a ROW; candidate j is a COLUMN. A need not be symmetric.
    with torch.autocast(device_type=v0.device.type, enabled=False):
        similarity = (unit @ unit.T).clamp(-1.0, 1.0)
    utilities = (similarity - similarity.mean(dim=1, keepdim=True)).clamp_min(0)
    if not bool(torch.isfinite(utilities).all()):
        raise NumericalIssue("utilities: nonfinite values")
    return utilities
