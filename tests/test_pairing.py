"""Offline identity regression contracts; written, not executed.

Fixtures use tiny byte strings and a mocked standard-library hasher. No real
checkpoint, GPU or Hub download is needed to execute these contracts later.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from v2_pruning.pairing import PairingError, verify_pairing


def fingerprint(path):
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model, vision = root / "renamed-model", root / "renamed-vision"
        model.mkdir()
        vision.mkdir()
        tower = "openai/clip-vit-large-patch14-336"
        model_files = {
            "config.json": json.dumps({"mm_vision_tower": tower}).encode(),
            "pytorch_model.bin.index.json": json.dumps({"weight_map": {
                "model.mm_projector.0.weight": "pytorch_model-00001-of-00001.bin"}}).encode(),
            "pytorch_model-00001-of-00001.bin": b"model-and-projector-weight",
            "tokenizer.model": b"fixture-tokenizer",
            "tokenizer_config.json": b"{}",
            "special_tokens_map.json": b"{}",
        }
        vision_files = {"config.json": b"{}", "preprocessor_config.json": b"{}",
                        "pytorch_model.bin": b"vision-weight-A"}

        def component(folder, files, repo):
            result = {"repo_id": repo, "revision": "a" * 40, "files": {}}
            for name, data in files.items():
                (folder / name).write_bytes(data)
                result["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            return result

        profile = {"id": "llava-v1.5-7b", "relationship": {
            "model_vision_tower": tower, "source_url": "https://example.invalid/fixed-config"},
            "model": component(model, model_files, "liuhaotian/llava-v1.5-7b"),
            "vision": component(vision, vision_files, tower)}
        registry = root / "pairing_profiles.json"
        registry.write_text(json.dumps({"schema_version": 1, "profiles": [profile]}), encoding="utf-8")
        with mock.patch("v2_pruning.pairing._REGISTRY", registry), \
                mock.patch("v2_pruning.pairing._sha256_file", side_effect=fingerprint):
            yield model, vision, profile


class PairingTests(unittest.TestCase):
    def test_complete_pair_uses_files_not_directory_names(self):
        with fixture() as (model, vision, profile):
            report = verify_pairing(model, vision)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["profile_id"], profile["id"])
            self.assertEqual(report["original_mm_vision_tower"], profile["vision"]["repo_id"])
            self.assertEqual(report["model"]["revision"], "a" * 40)
            for kind in ("model", "vision"):
                for row in report[kind]["files"].values():
                    self.assertEqual(row["actual_sha256"], row["expected_sha256"])
                    self.assertEqual(row["actual_size"], row["expected_size"])
            self.assertIn("historical training revision", report["verification_scope"])

    def test_equal_size_different_vision_weights_are_rejected(self):
        with fixture() as (model, vision, _):
            (vision / "pytorch_model.bin").write_bytes(b"vision-weight-B")
            with self.assertRaises(PairingError) as caught:
                verify_pairing(model, vision)
            self.assertEqual(caught.exception.report["status"], "rejected")
            self.assertEqual(caught.exception.report["component"], "vision")
            self.assertEqual(caught.exception.report["file"], "pytorch_model.bin")
            self.assertNotEqual(caught.exception.report["actual_sha256"],
                                caught.exception.report["expected_sha256"])

    def test_equal_size_different_projector_shard_is_rejected(self):
        with fixture() as (model, vision, _):
            path = model / "pytorch_model-00001-of-00001.bin"
            original = path.read_bytes()
            path.write_bytes(b"X" + original[1:])
            with self.assertRaisesRegex(PairingError, "SHA256 mismatch"):
                verify_pairing(model, vision)

    def test_wrong_explicit_profile_or_unregistered_profile_is_rejected(self):
        with fixture() as (model, vision, profile):
            other = deepcopy(profile)
            other["id"] = "llava-v1.5-13b"
            other["model"]["files"]["config.json"]["sha256"] = "0" * 64
            with mock.patch("v2_pruning.pairing._load_profiles", return_value=[profile, other]):
                with self.assertRaisesRegex(PairingError, "config.json does not match"):
                    verify_pairing(model, vision, "llava-v1.5-13b")
            with self.assertRaisesRegex(PairingError, "Unknown pairing profile"):
                verify_pairing(model, vision, "self-declared")

    def test_original_vision_identity_must_agree_with_registered_relationship(self):
        with fixture() as (model, vision, profile):
            path = model / "config.json"
            path.write_text(json.dumps({"mm_vision_tower": "other/clip"}), encoding="utf-8")
            size, digest = fingerprint(path)
            profile["model"]["files"]["config.json"] = {"size": size, "sha256": digest}
            with mock.patch("v2_pruning.pairing._load_profiles", return_value=[profile]):
                with self.assertRaisesRegex(PairingError, "Original mm_vision_tower"):
                    verify_pairing(model, vision)

    def test_unreadable_registry_is_rejected_with_context(self):
        for contents, error_type in ((None, "FileNotFoundError"),
                                     (b"{broken", "JSONDecodeError"),
                                     (b"\xff", "UnicodeDecodeError")):
            with self.subTest(error_type=error_type), fixture() as (model, vision, _):
                registry = model.parent / "pairing_profiles.json"
                if contents is None:
                    registry.unlink()
                else:
                    registry.write_bytes(contents)
                with self.assertRaises(PairingError) as caught:
                    verify_pairing(model, vision)
                self.assertEqual(caught.exception.report["component"], "registry")
                self.assertEqual(caught.exception.report["path"], str(registry))
                self.assertEqual(caught.exception.report["error_type"], error_type)

    def test_directory_enumeration_failure_is_rejected_with_context(self):
        for kind in ("model", "vision"):
            with self.subTest(component=kind), fixture() as (model, vision, _):
                blocked = (model if kind == "model" else vision).resolve()
                original_iterdir = Path.iterdir

                def entries(path):
                    if path == blocked:
                        raise PermissionError("fixture: directory listing denied")
                    yield from original_iterdir(path)

                with mock.patch.object(Path, "iterdir", autospec=True, side_effect=entries):
                    with self.assertRaises(PairingError) as caught:
                        verify_pairing(model, vision)
                self.assertEqual(caught.exception.report["component"], kind)
                self.assertEqual(caught.exception.report["path"], str(blocked))
                self.assertEqual(caught.exception.report["error_type"], "PermissionError")

    def test_registry_programming_error_is_not_reclassified(self):
        with fixture() as (model, vision, _):
            with mock.patch("v2_pruning.pairing.json.load", side_effect=TypeError("fixture: programming error")):
                with self.assertRaisesRegex(TypeError, "programming error"):
                    verify_pairing(model, vision)

    def test_extra_loading_candidates_cannot_override_verified_files(self):
        for name in ("pytorch_model.bin", "model.safetensors", "added_tokens.json", "tokenizer.json",
                     "adapter_config.json", "PYTORCH_MODEL.BIN", "ADAPTER_CONFIG.JSON"):
            with self.subTest(name=name), fixture() as (model, vision, _):
                (model / name).write_bytes(b"unverified replacement")
                with self.assertRaisesRegex(PairingError, "Unregistered loading candidate"):
                    verify_pairing(model, vision)

    def test_case_normalized_duplicate_candidates_are_rejected(self):
        with fixture() as (model, vision, _):
            original = model / "pytorch_model-00001-of-00001.bin"
            uppercase = original.with_name(original.name.upper())
            if uppercase.exists():
                self.skipTest("Filesystem aliases differently cased filenames")
            uppercase.write_bytes(original.read_bytes())
            with self.assertRaisesRegex(PairingError, "Ambiguous case-normalized loading candidates"):
                verify_pairing(model, vision)

    def test_case_sensitive_host_does_not_rewrite_registered_file_paths(self):
        with fixture() as (model, vision, _):
            original = vision / "pytorch_model.bin"
            uppercase = original.with_name(original.name.upper())
            if uppercase.exists():
                self.skipTest("Filesystem aliases differently cased filenames")
            original.rename(uppercase)
            with self.assertRaises(PairingError) as caught:
                verify_pairing(model, vision)
            self.assertEqual(caught.exception.report["file"], "pytorch_model.bin")
            self.assertEqual(caught.exception.report["component"], "vision")

    def test_missing_file_is_rejected_with_context(self):
        with fixture() as (model, vision, _):
            (vision / "pytorch_model.bin").unlink()
            with self.assertRaises(PairingError) as caught:
                verify_pairing(model, vision)
            self.assertEqual(caught.exception.report["file"], "pytorch_model.bin")
            self.assertEqual(caught.exception.report["component"], "vision")

    def test_unused_official_extras_are_allowed(self):
        with fixture() as (model, vision, _):
            (model / "README.md").write_text("fixture", encoding="utf-8")
            (model / "mm_projector.bin").write_bytes(b"unused standalone projector")
            (vision / "tokenizer.json").write_text("{}", encoding="utf-8")
            self.assertEqual(verify_pairing(model, vision)["status"], "verified")

    def test_external_snapshot_blob_symlinks_are_allowed(self):
        with fixture() as (model, vision, _):
            path = vision / "pytorch_model.bin"
            blob = vision.parent / "fixture-blob"
            blob.write_bytes(path.read_bytes())
            path.unlink()
            try:
                path.symlink_to(blob)
            except OSError as exc:
                self.skipTest(f"Host does not permit symlinks: {exc}")
            self.assertEqual(verify_pairing(model, vision)["status"], "verified")


if __name__ == "__main__":
    unittest.main()
