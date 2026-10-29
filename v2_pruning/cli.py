"""Offline V2 inference entry point; nothing runs unless main() is invoked."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

from .config import PruningConfig
from .pairing import PairingError


def _reject_json_constant(value: str):
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2 dynamic visual-token pruning, offline original LLaVA-1.5")
    parser.add_argument("--model-path", type=Path, required=True, help="Local original LLaVA-1.5 checkpoint directory")
    parser.add_argument("--vision-path", type=Path, required=True, help="Local complete CLIP vision checkpoint directory")
    parser.add_argument("--pairing-profile", choices=("auto", "llava-v1.5-7b", "llava-v1.5-13b"), default="auto",
                        help="Verify pinned official model/CLIP identities before loading; auto matches model config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--question-file", type=Path, required=True, help="Local JSONL with image and question/text fields")
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quality-protocol", choices=("exact_match", "none"), default="exact_match",
                        help="DEMO answer match only; not a dataset's official VQA metric")
    return parser


def read_questions(path: Path, image_folder: Path, limit: int | None = None) -> list[dict]:
    """Accept upstream AgilePruner JSONL fields without its random answer IDs."""
    root = image_folder.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("image-folder must be a local directory")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    records = []
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(item, dict):
                raise ValueError(f"Question line {line_no} must be an object")
            question = item.get("question", item.get("text"))
            image = item.get("image")
            if not isinstance(question, str):
                raise ValueError(f"Question line {line_no} requires a string question/text")
            if not isinstance(image, str) or not image or "://" in image:
                raise ValueError(f"Question line {line_no} requires a local image filename")
            # Keep filesystem failures local to the sample-processing loop.
            image_path = root / image
            identifier = item.get("question_id", item.get("id", line_no))
            if (isinstance(identifier, bool) or not isinstance(identifier, (str, int))
                    or (isinstance(identifier, str) and not identifier.strip())):
                raise ValueError(f"Question line {line_no} requires an integer or nonempty string id")
            identity = json.dumps(identifier, sort_keys=True, ensure_ascii=False)
            if identity in seen:
                raise ValueError(f"Duplicate question id on line {line_no}")
            seen.add(identity)
            records.append({
                "question_id": identifier, "question": question,
                "image_path": str(image_path), "image": image,
                "reference_answer": item.get("reference_answer", item.get("answer", item.get("answers"))),
            })
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError("The question file is empty")
    return records


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_image_path(path: Path, image_folder: Path) -> Path:
    """Resolve and validate one sample, including symlinks, at processing time."""
    root = image_folder.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("image must be a local file within image-folder")
    return resolved


def _load_runtime():
    """All optional imports happen before an output directory is claimed."""
    from .llava_adapter import LlavaAdapter, enforce_offline, preflight_runtime
    enforce_offline()
    preflight_runtime()
    import torch
    import transformers
    import huggingface_hub
    from PIL import Image
    from .probe import ProbeUnavailable, probe_relevance
    from .selector import prune_visual_tokens
    from .types import PruningResult, SelectionResult
    from .reporting import write_case
    from .evaluation import score_answer, summarize_records
    return SimpleNamespace(
        torch=torch, transformers=transformers, hub_version=huggingface_hub.__version__,
        Image=Image, LlavaAdapter=LlavaAdapter,
        ProbeUnavailable=ProbeUnavailable, probe_relevance=probe_relevance,
        prune_visual_tokens=prune_visual_tokens, PruningResult=PruningResult,
        SelectionResult=SelectionResult, write_case=write_case,
        score_answer=score_answer, summarize_records=summarize_records,
    )


def input_audit(prepared, selected: list[int]) -> dict:
    """Persist the actual unpruned Probe layout and the gathered native LM layout."""
    visual = set(prepared.visual_positions)
    retained_visual = {prepared.visual_positions[i] for i in selected}
    length = int(prepared.token_ids.shape[1])
    retained_sequence = [i for i in range(length) if i not in visual or i in retained_visual]
    return {
        "prompt": prepared.prompt,
        "question_positions": list(prepared.question_positions),
        "visual_positions": list(prepared.visual_positions),
        "probe_position_ids": prepared.position_ids.detach().cpu().tolist(),
        "prompt_token_ids_with_visual_placeholders": prepared.token_ids.detach().cpu().tolist(),
        "token_id_note": "Visual slots contain bookkeeping placeholder IDs; model input uses original V0 embeddings",
        "retained_original_visual_indices": list(selected),
        "retained_original_sequence_indices": retained_sequence,
        "generation_position_ids": [list(range(len(retained_sequence)))],
        "index_convention": "0-based; question/visual positions refer to the full unpruned Probe sequence",
    }


def fallback_reason(prepared, question: str) -> str | None:
    """Skip pruning when its required question or spatial mapping is unavailable."""
    reasons = []
    if not question.strip():
        reasons.append("no effective question instruction")
    if prepared.grid_shape is None:
        reasons.append(prepared.grid_error or "visual spatial mapping is unavailable")
    return "; ".join(reasons) if reasons else None


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    cfg = PruningConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    questions = read_questions(args.question_file, args.image_folder, args.limit)
    # Missing imports never leave behind an empty, non-retryable output directory.
    try:
        runtime = _load_runtime()
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("Runtime dependencies could not be imported; output directory was not created. "
                           f"Fix the existing environment and retry the same command: {exc}") from exc
    torch, transformers, Image = runtime.torch, runtime.transformers, runtime.Image
    LlavaAdapter, ProbeUnavailable = runtime.LlavaAdapter, runtime.ProbeUnavailable
    probe_relevance, prune_visual_tokens = runtime.probe_relevance, runtime.prune_visual_tokens
    PruningResult, SelectionResult = runtime.PruningResult, runtime.SelectionResult
    write_case, score_answer, summarize_records = runtime.write_case, runtime.score_answer, runtime.summarize_records
    # Fail before model loading if output already exists; never overwrite prior evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "implementation": "V2 region coverage and dynamic stopping",
        "status": "loading", "config": asdict(cfg), "seed": args.seed,
        "model_path": str(args.model_path.resolve()), "vision_path": str(args.vision_path.resolve()),
        "pairing_profile_requested": args.pairing_profile,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "huggingface_hub": runtime.hub_version,
        "python": platform.python_version(), "device": args.device,
        "conv_mode": args.conv_mode, "max_new_tokens": args.max_new_tokens,
        "dtype": "float16 model; float32 coverage/probe statistics",
        "position_policy": "native_compact", "batch_size": 1, "decoding": "greedy, one beam",
        "runtime_downloads": False, "model_loading_in_timing": False,
        "timing_scope": "image open + preprocessing + vision/projector + probe + selection + generation; excludes report export",
        "timing_note": "single observations without warmup; synchronized stage/decoder profiling perturbs latency; "
                       "nested stages overlap parent totals and must not be summed; not a benchmark or speedup claim",
        "baseline_reference": "AgilePruner published results; no baseline is executed by this entry point",
        "quality_protocol": args.quality_protocol,
        "quality_note": "DEMO exact match is not official VQA/AgilePruner paper accuracy",
    }
    _write_json(args.output_dir / "run.json", manifest)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Numerical sums/similarities should not silently use TF32 on Ampere cards.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = LlavaAdapter(args.model_path.resolve(), args.vision_path.resolve(), args.device,
                               pairing_profile=args.pairing_profile)
    except Exception as exc:
        manifest.update(status="failed_identity" if isinstance(exc, PairingError) else "failed_loading",
                        error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, PairingError):
            manifest["pairing_identity"] = exc.report
        _write_json(args.output_dir / "run.json", manifest)
        raise
    manifest.update(status="running", gpu=torch.cuda.get_device_name(adapter.device),
                    checkpoint_loading=adapter.loading_info, pairing_identity=adapter.pairing_info)
    _write_json(args.output_dir / "run.json", manifest)
    completed = failures = 0
    status_counts: dict[str, int] = {}
    completed_records, error_records = [], []

    def save_summary():
        _write_json(args.output_dir / "summary.json", summarize_records(
            completed_records, error_records, len(questions), args.quality_protocol
        ))

    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as results, \
            (args.output_dir / "errors.jsonl").open("w", encoding="utf-8") as errors, \
            (args.output_dir / "answers_agile_format.jsonl").open("w", encoding="utf-8") as answers:
        for index, item in enumerate(questions, 1):
            try:
                adapter.synchronize()
                begin = time.perf_counter()
                image_path = resolve_image_path(Path(item["image_path"]), args.image_folder)
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                prepared = adapter.prepare(image, item["question"], args.conv_mode)
                adapter.synchronize()
                prepared_at = time.perf_counter()
                raw = None
                reason = fallback_reason(prepared, item["question"])
                try:
                    if reason is None:
                        raw = probe_relevance(
                            adapter.model, prepared.inputs_embeds, prepared.question_positions,
                            prepared.visual_positions, prepared.position_ids, cfg.query_chunk_size,
                        )
                except ProbeUnavailable as exc:
                    reason = f"probe: {exc}"
                if reason is not None:
                    selection = SelectionResult(
                        list(range(len(prepared.visual_positions))), [], "fallback_full", reason, []
                    )
                    pruned = PruningResult(selection, None, None, None)
                adapter.synchronize()
                probed_at = time.perf_counter()
                pruning_timings = {}
                if raw is not None:
                    pruned = prune_visual_tokens(prepared.v0, raw, prepared.grid_shape, cfg,
                                                 timings=pruning_timings)
                adapter.synchronize()
                selected_at = time.perf_counter()
                generation = adapter.generate(prepared, pruned.selection.indices, args.max_new_tokens)
                adapter.synchronize()
                end = time.perf_counter()
                selection = pruned.selection
                n, k = len(prepared.visual_positions), len(selection.indices)
                ttft = generation.pop("generation_ttft_ms")
                generation_ms = generation.pop("generation_ms")
                prefill_ms = generation.pop("generation_prefill_ms", None)
                decode_ms = generation.pop("generation_decode_forward_ms", None)
                generation.pop("generation_decode_ms", None)
                record = {
                    **item, **generation, "config": asdict(cfg), "selection": asdict(selection),
                    "visual_tokens_original": n, "visual_tokens_kept": k,
                    "retained_fraction": k / n if n else None,
                    "region_count": len(pruned.regions.regions) if pruned.regions else None,
                    "grid_shape": None if prepared.grid_shape is None else list(prepared.grid_shape),
                    "spatial_mapping_error": prepared.grid_error,
                    "quality": score_answer(generation["answer"], item["reference_answer"], args.quality_protocol),
                    "input_audit": input_audit(prepared, selection.indices),
                    "timing_ms": {
                        "input_vision_projector": (prepared_at - begin) * 1000,
                        "probe": (probed_at - prepared_at) * 1000,
                        "regions_coverage_selection": (selected_at - probed_at) * 1000,
                        "generation": generation_ms,
                        "ttft": None if ttft is None else (selected_at - begin) * 1000 + ttft,
                        "end_to_end": (end - begin) * 1000,
                        "preprocessing": (prepared.stage_timings_ms or {}).get("preprocessing"),
                        "vision_encoder": (prepared.stage_timings_ms or {}).get("vision_encoder"),
                        "projector": (prepared.stage_timings_ms or {}).get("projector"),
                        "text_input": (prepared.stage_timings_ms or {}).get("text_input"),
                        "signal_weights": pruning_timings.get("signal_weights"),
                        "regions": pruning_timings.get("regions"),
                        "coverage_matrix": pruning_timings.get("coverage_matrix"),
                        "selection": pruning_timings.get("selection"),
                        "llm_prefill_forward": prefill_ms,
                        "llm_decode_forward": decode_ms,
                    },
                    "trace_scope": "attempted selection; fallback_full returns all original tokens",
                    "original_image_size": list(prepared.original_size),
                    "raw_relevance": None if raw is None else raw.detach().cpu().tolist(),
                    "q": None if pruned.q is None else pruned.q.detach().cpu().tolist(),
                    "w": None if pruned.w is None else pruned.w.detach().cpu().tolist(),
                    "region_labels": None if pruned.regions is None else pruned.regions.labels.cpu().tolist(),
                    "region_seeds": None if pruned.regions is None else pruned.regions.seeds,
                }
                paths = write_case(
                    args.output_dir / f"case_{index:06d}", record, adapter.canvas(prepared),
                    record["q"], record["region_labels"], selection.indices, prepared.grid_shape,
                )
                # Re-read the JSON writer's finite representation for the shared JSONL.
                safe_record = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
                results.write(json.dumps(safe_record, ensure_ascii=False, allow_nan=False) + "\n")
                results.flush()
                # Match upstream model_vqa_loader's answer schema for later official scoring.
                answers.write(json.dumps({
                    "question_id": item["question_id"], "prompt": item["question"],
                    "text": generation["answer"], "answer_id": f"v2-{index:06d}",
                    "model_id": args.model_path.name,
                    "metadata": {"visual_tokens": k, "selection_status": selection.status,
                                 "position_policy": "native_compact"},
                }, ensure_ascii=False, allow_nan=False) + "\n")
                answers.flush()
                completed_records.append(safe_record)
                completed += 1
                status_counts[selection.status] = status_counts.get(selection.status, 0) + 1
                print(f"{index}/{len(questions)}  {selection.status}  K={k}/{n}")
            except torch.cuda.OutOfMemoryError:
                # OOM is an execution failure, never a successful full-keep fallback.
                manifest.update(status="failed_oom", completed=completed, failed=failures + 1)
                _write_json(args.output_dir / "run.json", manifest)
                error_records.append({"question_id": item["question_id"], "error_type": "OutOfMemoryError"})
                errors.write(json.dumps(error_records[-1], ensure_ascii=False) + "\n")
                errors.flush()
                save_summary()
                raise
            except (ValueError, OSError, RuntimeError, Image.DecompressionBombError) as exc:
                failures += 1
                error = {"question_id": item["question_id"], "image": item["image"],
                         "error_type": type(exc).__name__, "error": str(exc)}
                error_records.append(error)
                errors.write(json.dumps(error, ensure_ascii=False, allow_nan=False) + "\n")
                errors.flush()
    save_summary()
    manifest.update(status="finished" if not failures else "finished_with_errors",
                    completed=completed, failed=failures, total=len(questions), selection_status_counts=status_counts)
    _write_json(args.output_dir / "run.json", manifest)
    if failures:
        raise SystemExit(1)
