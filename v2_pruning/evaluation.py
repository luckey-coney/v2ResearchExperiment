"""DEMO exact match and descriptive single-run statistics, using only stdlib.

This exact-match protocol is NOT official VQA accuracy. Official dataset scores
must be supplied separately with their evaluator protocol and source provenance.
No model execution, answer extraction, or fallback trace interpretation occurs.
"""
from __future__ import annotations

from collections import Counter
import math
import unicodedata


def _protocol(protocol: str) -> None:
    if protocol not in ("exact_match", "none"):
        raise ValueError("Supported quality protocols: exact_match (DEMO), none")


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _references(reference: object) -> tuple[list[str], str | None]:
    if reference is None:
        return [], "missing_reference"
    values = reference if isinstance(reference, list) else [reference]
    if not values:
        return [], "empty_reference_list"
    normalized = []
    for value in values:
        if isinstance(value, str):
            answer = _normalize(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            answer = str(value)
        elif isinstance(value, float) and math.isfinite(value):
            answer = str(value)
        else:
            return [], "unsupported_or_nonfinite_reference"
        if answer:
            normalized.append(answer)
    return (normalized, None) if normalized else ([], "empty_reference")


def score_answer(prediction: str, reference: object, protocol: str = "exact_match") -> dict:
    """Return DEMO any-reference exact match; unlabeled samples remain unscored.

    Normalize Unicode NFKC, case and whitespace only; preserve punctuation and
    complete answers. No article/number conversion or VQA consensus is applied.
    Unsupported reference structures are unscored, never silently misread.
    """
    _protocol(protocol)
    references, reason = _references(reference)
    result = {"protocol": protocol, "score": None,
              "reference_count": len(references), "reason": None}
    if protocol == "none":
        result["reason"] = "none: scoring_disabled" + (f"; {reason}" if reason else "")
    elif reason:
        result["reason"] = f"exact_match: {reason}"
    elif not isinstance(prediction, str):
        result["reason"] = "exact_match: prediction_must_be_string"
    else:
        result["score"] = float(_normalize(prediction) in references)
    return result


def _distribution(values) -> dict:
    values = sorted(v for v in values if _finite(v) and v >= 0)
    result = dict.fromkeys(("mean", "median", "p90", "p95", "min", "max"))
    result["count"] = len(values)
    if values:
        def quantile(q):
            position = (len(values) - 1) * q
            low, high = math.floor(position), math.ceil(position)
            return values[low] + (values[high] - values[low]) * (position - low)
        result.update(mean=math.fsum(v / len(values) for v in values),
                      median=quantile(0.5), p90=quantile(0.9), p95=quantile(0.95),
                      min=values[0], max=values[-1])
    return result


def _rate(count: int, denominator: int) -> dict:
    return {"count": count, "denominator": denominator,
            "rate": count / denominator if denominator else None}


def _metrics(records: list[dict]) -> dict:
    pairs = [(r.get("visual_tokens_kept"), r.get("visual_tokens_original")) for r in records]
    pairs = [(k, n) for k, n in pairs
             if _finite(k) and _finite(n) and 0 <= k <= n and n > 0]
    kept, original = sum(k for k, _ in pairs), sum(n for _, n in pairs)
    timing_keys = sorted({key for r in records if isinstance(r.get("timing_ms"), dict)
                          for key in r["timing_ms"] if isinstance(key, str)})
    timing = {key: _distribution(r.get("timing_ms", {}).get(key) for r in records
                                if isinstance(r.get("timing_ms"), dict)) for key in timing_keys}
    return {
        "visual_tokens_kept": _distribution(r.get("visual_tokens_kept") for r in records),
        "compression": {"valid_pair_count": len(pairs), "denominator_original_tokens": original,
                        "total_kept_tokens": kept,
                        "retained_fraction_weighted": kept / original if original else None,
                        "reduction_fraction_weighted": 1 - kept / original if original else None,
                        "zero_compression_count": sum(k == n for k, n in pairs)},
        "region_count": _distribution(r.get("region_count") for r in records),
        "generated_tokens": _distribution(r.get("generated_tokens") for r in records),
        "timing_ms": timing,
    }


def summarize_records(records: list[dict], errors: list[dict], total: int,
                      protocol: str = "exact_match") -> dict:
    """Aggregate completed records without dropping difficult full-keep fallbacks.

    Failed/unprocessed samples have no invented prediction or quality score.
    Selection status and actual retained counts are used; traces are not read.
    Every distribution reports its own finite nonnegative observation count.
    """
    _protocol(protocol)
    if (isinstance(total, bool) or not isinstance(total, int) or total < 0
            or len(records) + len(errors) > total):
        raise ValueError("total must cover all completed and failed records")
    completed, failed = len(records), len(errors)
    scores, reasons, without_reference = [], Counter(), 0
    groups: dict[str, list[dict]] = {}
    group_scores, group_unlabeled = {}, Counter()
    for record in records:
        selection = record.get("selection")
        status = selection.get("status") if isinstance(selection, dict) else None
        status = status if isinstance(status, str) and status else "unknown"
        groups.setdefault(status, []).append(record)
        group_scores.setdefault(status, [])
        quality = record.get("quality")
        quality = quality if isinstance(quality, dict) else {}
        count = quality.get("reference_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            count = len(_references(record.get("reference_answer"))[0])
        without_reference += count == 0
        group_unlabeled[status] += count == 0
        score = quality.get("score")
        if (protocol == "exact_match" and quality.get("protocol") == protocol
                and count > 0 and _finite(score) and 0 <= score <= 1):
            scores.append(score)
            group_scores[status].append(score)
        else:
            reason = ("none: scoring_disabled" if protocol == "none" else
                      f"{protocol}: protocol_mismatch" if quality.get("protocol") != protocol else
                      quality.get("reason") or f"{protocol}: missing_or_invalid_score")
            reasons[str(reason)] += 1
    metrics = _metrics(records)
    statuses = {status: len(group) for status, group in sorted(groups.items())}
    selection = {"status_counts": statuses}
    for status in ("fallback_full", "satisfied", "resource_limited_unsatisfied"):
        selection[status] = _rate(statuses.get(status, 0), completed)
    compression = metrics["compression"]
    selection["zero_compression"] = _rate(compression["zero_compression_count"],
                                           compression["valid_pair_count"])
    return {
        "counts": {"total": total, "completed": completed, "failed": failed,
                   "unprocessed": total - completed - failed},
        "quality": {"protocol": protocol, "metric_label": "DEMO exact_match" if protocol == "exact_match" else "disabled",
                    "scored": len(scores), "score_denominator": len(scores),
                    "unscored_completed": completed - len(scores), "without_reference": without_reference,
                    "accuracy_scored_only": math.fsum(scores) / len(scores) if scores else None,
                    "coverage_scored_over_completed": len(scores) / completed if completed else None,
                    "coverage_scored_over_total": len(scores) / total if total else None,
                    "reference_coverage_over_completed": (completed - without_reference) / completed if completed else None,
                    "unscored_reasons": dict(sorted(reasons.items())),
                    "normalization": "Unicode NFKC + casefold + collapse whitespace; punctuation preserved; no answer extraction",
                    "note": "DEMO exact_match is not official VQA accuracy; none disables scoring; without_reference means no usable reference",
                    "external_official_scoring": {"status": "not_run", "protocol": None, "source": None}},
        **metrics, "selection": selection,
        "by_selection_status": {status: {"count": len(group), **_metrics(group),
                                "quality": {"score_denominator": len(group_scores[status]),
                                            "unscored_completed": len(group) - len(group_scores[status]),
                                            "without_reference": group_unlabeled[status],
                                            "accuracy_scored_only": (math.fsum(group_scores[status]) / len(group_scores[status])
                                                                     if group_scores[status] else None)}}
                                for status, group in sorted(groups.items())},
        "quantile_method": "linear interpolation at (count - 1) * q on sorted observations",
        "timing_note": "single-run observations without warmup; synchronized profiling perturbs latency; "
                       "nested stages overlap parent totals; descriptive only, not a benchmark or speedup claim",
        "selection_note": "all completed samples included; fallback uses actual K=N, never attempted-trace coverage",
    }
