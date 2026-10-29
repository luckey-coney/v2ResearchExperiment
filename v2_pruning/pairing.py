"""Offline publisher-file identity checks for registered LLaVA/CLIP pairs.

File hashing reuses huggingface_hub.utils.sha.sha_fileobj; see the source and
license registry. No repository names supplied by a local checkpoint are trusted
as evidence on their own, and no network or self-enrollment path exists here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat


_REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "pairing_profiles.json"
_PROFILE_IDS = ("llava-v1.5-7b", "llava-v1.5-13b")
_MODEL_RUNTIME_NAMES = {
    "config.json", "generation_config.json", "tokenizer.model", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "vocab.txt", "merges.txt", "chat_template.jinja", "processor_config.json",
}
_VISION_RUNTIME_NAMES = {"config.json", "preprocessor_config.json", "processor_config.json"}
_ADAPTER_RUNTIME_NAMES = {"adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"}


class PairingError(ValueError):
    """Unverified identity is an initialization failure, never pruning fallback."""
    def __init__(self, message: str, **details):
        super().__init__(message)
        self.report = {"status": "rejected", "reason": message, **details}
        self.details = self.report


def _sha256_file(path: Path) -> tuple[int, str]:
    # Direct reuse from the installed Hub dependency; no copied hashing loop.
    from huggingface_hub.utils.sha import sha_fileobj

    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PairingError(f"Expected a regular local file: {path}", path=str(path))
        digest = sha_fileobj(stream).hex()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PairingError(f"File changed while hashing: {path}", path=str(path))
    return before.st_size, digest


def _load_profiles() -> list[dict]:
    try:
        with _REGISTRY.open(encoding="utf-8-sig") as stream:
            registry = json.load(stream)
    except (OSError, ValueError) as exc:
        raise PairingError(f"Cannot read built-in pairing registry: {exc}",
                           component="registry", path=str(_REGISTRY),
                           error_type=type(exc).__name__) from exc
    if (not isinstance(registry, dict) or registry.get("schema_version") != 1
            or not isinstance(registry.get("profiles"), list) or not registry["profiles"]):
        raise PairingError("Invalid built-in pairing registry", path=str(_REGISTRY))
    return registry["profiles"]


def _validate_profile(profile: dict) -> None:
    if not isinstance(profile, dict) or profile.get("id") not in _PROFILE_IDS:
        raise PairingError("Invalid built-in pairing profile")
    for kind in ("model", "vision"):
        component = profile.get(kind)
        if (not isinstance(component, dict) or not isinstance(component.get("repo_id"), str)
                or not component["repo_id"] or not isinstance(component.get("revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", component["revision"])
                or not isinstance(component.get("files"), dict) or "config.json" not in component["files"]):
            raise PairingError(f"Invalid built-in {kind} identity", profile_id=profile["id"])
        for name, expected in component["files"].items():
            if (not isinstance(name, str) or not name or name in (".", "..")
                    or any(character in name for character in "/\\:")
                    or not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
                    or isinstance(expected.get("size"), bool) or not isinstance(expected.get("size"), int)
                    or expected["size"] < 0):
                raise PairingError(f"Invalid built-in {kind} file identity", profile_id=profile["id"], file=name)
    relationship = profile.get("relationship")
    if (not isinstance(relationship, dict)
            or relationship.get("model_vision_tower") != profile["vision"]["repo_id"]):
        raise PairingError("Built-in model/vision relationship is inconsistent", profile_id=profile["id"])


def _directory(value: Path, kind: str) -> Path:
    path = Path(value)
    try:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise PairingError(f"Expected local {kind} directory: {path}", component=kind, path=str(path))
    except OSError as exc:
        raise PairingError(f"Cannot access local {kind} directory: {path}: {exc}",
                           component=kind, path=str(path), error_type=type(exc).__name__) from exc
    return path


def _fingerprint(path: Path, kind: str, expected: dict | None = None) -> tuple[int, str]:
    details = {"component": kind, "file": path.name, "path": str(path)}
    if expected is not None:
        details.update(expected_sha256=expected["sha256"], expected_size=expected["size"])
    try:
        if not path.is_file():
            raise PairingError(f"Required {kind} file is missing or not a regular file: {path}", **details)
        actual_size = path.stat().st_size
        if expected is not None and actual_size != expected["size"]:
            raise PairingError(f"Size mismatch for {kind}/{path.name}", actual_size=actual_size, **details)
        size, digest = _sha256_file(path)
    except OSError as exc:
        raise PairingError(f"Cannot read {kind}/{path.name}: {exc}",
                           error_type=type(exc).__name__, **details) from exc
    if expected is not None and (size != expected["size"] or digest != expected["sha256"]):
        raise PairingError(f"SHA256 mismatch for {kind}/{path.name}",
                           actual_size=size, actual_sha256=digest, **details)
    return size, digest


def _reject_unregistered_candidates(root: Path, component: dict, kind: str) -> None:
    expected = {name.casefold() for name in component["files"]}
    runtime_names = (_MODEL_RUNTIME_NAMES if kind == "model" else _VISION_RUNTIME_NAMES) | _ADAPTER_RUNTIME_NAMES
    seen = {}
    try:
        entries = list(root.iterdir())
    except (OSError, ValueError) as exc:
        raise PairingError(f"Cannot list local {kind} directory: {root}: {exc}",
                           component=kind, path=str(root), error_type=type(exc).__name__) from exc
    for entry in entries:
        # Windows loaders may resolve uppercase aliases of conventional names.
        # Recognize those aliases on every host; do not rewrite actual paths.
        name = entry.name.casefold()
        weight_candidate = (name.startswith("pytorch_model") and
                            (name.endswith(".bin") or name.endswith(".bin.index.json"))) or (
                            name.startswith("model") and
                            (name.endswith(".safetensors") or name.endswith(".safetensors.index.json")))
        if not (weight_candidate or name in runtime_names):
            continue
        if name in seen:
            raise PairingError(f"Ambiguous case-normalized loading candidates in {kind}: "
                               f"{seen[name]} and {entry.name}", component=kind, file=entry.name,
                               conflicting_file=seen[name], path=str(entry))
        seen[name] = entry.name
        if name not in expected:
            raise PairingError(f"Unregistered loading candidate in {kind}: {name}",
                               component=kind, file=entry.name, path=str(entry))


def _verify_component(root: Path, component: dict, kind: str) -> dict:
    _reject_unregistered_candidates(root, component, kind)
    files = {}
    for name, expected in component["files"].items():
        path = root / name
        size, digest = _fingerprint(path, kind, expected)
        files[name] = {"path": str(path), "expected_size": expected["size"], "actual_size": size,
                       "expected_sha256": expected["sha256"], "actual_sha256": digest}
    return {"repo_id": component["repo_id"], "revision": component["revision"],
            "local_path": str(root), "weight_format": "pytorch_bin", "files": files}


def verify_pairing(model_path: Path, vision_path: Path, profile: str = "auto") -> dict:
    """Verify registered publisher bytes before loading any checkpoint weights.

    Normal Hugging Face snapshot symlinks are accepted. Names of local folders
    are irrelevant; every startup hashes actual bytes. Unknown/repacked/modified
    checkpoints require a separately reviewed built-in profile, never enrollment
    from their own current hashes. Callers must load PyTorch bin explicitly.
    """
    if profile not in ("auto", *_PROFILE_IDS):
        raise PairingError(f"Unknown pairing profile: {profile}", requested_profile=profile)
    model_root = _directory(model_path, "model")
    vision_root = _directory(vision_path, "vision")
    profiles = _load_profiles()
    for candidate in profiles:
        _validate_profile(candidate)
    if len({candidate["id"] for candidate in profiles}) != len(profiles):
        raise PairingError("Duplicate built-in pairing profile IDs")
    config_size, config_hash = _fingerprint(model_root / "config.json", "model")
    candidates = [candidate for candidate in profiles
                  if (profile == "auto" or candidate["id"] == profile)
                  and candidate["model"]["files"]["config.json"]["sha256"] == config_hash
                  and candidate["model"]["files"]["config.json"]["size"] == config_size]
    if len(candidates) != 1:
        raise PairingError("Local model config.json does not match exactly one registered pairing profile",
                           requested_profile=profile, component="model", file="config.json",
                           actual_size=config_size, actual_sha256=config_hash,
                           registered_profiles=[candidate["id"] for candidate in profiles])
    selected = candidates[0]
    try:
        original_config = json.loads((model_root / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise PairingError(f"Cannot read original model config: {exc}",
                           profile_id=selected["id"], component="model", file="config.json") from exc
    original_tower = original_config.get("mm_vision_tower") if isinstance(original_config, dict) else None
    expected_tower = selected["relationship"]["model_vision_tower"]
    if original_tower != expected_tower:
        raise PairingError("Original mm_vision_tower disagrees with registered paired vision identity",
                           profile_id=selected["id"], expected_mm_vision_tower=expected_tower,
                           original_mm_vision_tower=original_tower)
    model_info = _verify_component(model_root, selected["model"], "model")
    vision_info = _verify_component(vision_root, selected["vision"], "vision")
    registry_size, registry_hash = _fingerprint(_REGISTRY, "registry")
    return {
        "status": "verified", "profile_id": selected["id"],
        "original_mm_vision_tower": original_tower,
        "relationship_source": selected["relationship"].get("source_url"),
        "registry_path": str(_REGISTRY), "registry_sha256": registry_hash, "registry_size": registry_size,
        "model": model_info, "vision": vision_info,
        "verification_scope": "Local files match pinned publisher repository bytes and the model-config-declared "
                              "vision repository; this does not establish the historical training revision "
                              "of the vision encoder or task quality. Files must remain unchanged during loading.",
    }
