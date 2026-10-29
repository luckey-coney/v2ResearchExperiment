"""Offline pairing entry-point contracts, written for later execution only."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest import TestCase, mock

from v2_pruning.cli import main
from v2_pruning.llava_adapter import LlavaAdapter
from v2_pruning.pairing import PairingError
from test_pairing import fixture


class PairingIntegrationTests(TestCase):
    def test_adapter_rejects_unverified_pair_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, vision = root / "model", root / "vision"
            for folder in (model, vision):
                folder.mkdir()
                (folder / "config.json").write_text("{}", encoding="utf-8")
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), device=lambda value: value)
            classes = [mock.Mock() for _ in range(5)]
            failure = PairingError("fixture: same structure but different vision bytes")
            with mock.patch("v2_pruning.llava_adapter.preflight_runtime", return_value=(fake_torch, *classes)), \
                    mock.patch("v2_pruning.llava_adapter.verify_pairing", side_effect=failure) as verify:
                with self.assertRaises(PairingError):
                    LlavaAdapter(model, vision, pairing_profile="llava-v1.5-7b")
            verify.assert_called_once_with(model, vision, "llava-v1.5-7b")
            for native_class in classes:
                native_class.from_pretrained.assert_not_called()

    def test_cli_records_rejected_identity_and_never_processes_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "config.json").write_text(json.dumps({"seed_threshold": .8, "grow_threshold": .5,
                 "tau": .3, "alpha_all": .75, "alpha_local": .9}), encoding="utf-8")
            (root / "questions.jsonl").write_text(json.dumps({"image": "unused.png", "question": "Count?"}) + "\n",
                                                   encoding="utf-8")
            output = root / "output"
            runtime = mock.MagicMock()
            runtime.torch.__version__ = "fixture"
            runtime.transformers.__version__ = "fixture"
            runtime.hub_version = "fixture"
            runtime.LlavaAdapter.side_effect = PairingError("fixture: unregistered vision identity")
            argv = ["--model-path", str(root / "model"), "--vision-path", str(root / "vision"),
                    "--config", str(root / "config.json"), "--question-file", str(root / "questions.jsonl"),
                    "--image-folder", str(root / "images"), "--output-dir", str(output)]
            with mock.patch("v2_pruning.cli._load_runtime", return_value=runtime):
                with self.assertRaises(PairingError):
                    main(argv)
            report = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed_identity")
            self.assertEqual(report["pairing_profile_requested"], "auto")
            self.assertEqual(report["error_type"], "PairingError")
            self.assertIn("unregistered vision", report["error"])
            runtime.probe_relevance.assert_not_called()
            runtime.Image.open.assert_not_called()
            self.assertFalse((output / "results.jsonl").exists())

    def test_cli_records_real_identity_input_failures_before_weight_loading(self):
        cases = (
            ("model", "directory", "FileNotFoundError"),
            ("vision", "directory", "FileNotFoundError"),
            ("model", "config", None),
            ("vision", "config", None),
            ("registry", "missing", "FileNotFoundError"),
            ("registry", "invalid_json", "JSONDecodeError"),
            ("model", "listing", "PermissionError"),
            ("vision", "listing", "PermissionError"),
        )
        for component, failure, error_type in cases:
            with self.subTest(component=component, failure=failure), fixture() as (model, vision, _):
                root = model.parent
                registry = root / "pairing_profiles.json"
                target = {"model": model, "vision": vision, "registry": registry}[component]
                failed_path = target / "config.json" if failure == "config" else target
                if failure == "directory":
                    target.rename(root / f"unavailable-{component}")
                elif failure in ("config", "missing"):
                    failed_path.unlink()
                elif failure == "invalid_json":
                    registry.write_text("{broken", encoding="utf-8")
                (root / "images").mkdir()
                (root / "config.json").write_text(json.dumps({"seed_threshold": .8, "grow_threshold": .5,
                    "tau": .3, "alpha_all": .75, "alpha_local": .9}), encoding="utf-8")
                (root / "questions.jsonl").write_text(
                    json.dumps({"image": "unused.png", "question": "Count?"}) + "\n", encoding="utf-8")
                output = root / "output"
                runtime = mock.MagicMock()
                runtime.torch.__version__ = runtime.transformers.__version__ = runtime.hub_version = "fixture"
                runtime.torch.cuda.is_available.return_value = True
                runtime.torch.device.side_effect = lambda value: value
                runtime.LlavaAdapter = LlavaAdapter
                native_classes = [mock.Mock() for _ in range(5)]
                original_iterdir = Path.iterdir

                def entries(path):
                    if path == target.resolve():
                        raise PermissionError("fixture: directory listing denied")
                    yield from original_iterdir(path)

                argv = ["--model-path", str(model), "--vision-path", str(vision),
                        "--config", str(root / "config.json"), "--question-file", str(root / "questions.jsonl"),
                        "--image-folder", str(root / "images"), "--output-dir", str(output)]
                with ExitStack() as stack:
                    stack.enter_context(mock.patch("v2_pruning.cli._load_runtime", return_value=runtime))
                    stack.enter_context(mock.patch("v2_pruning.llava_adapter.preflight_runtime",
                                                   return_value=(runtime.torch, *native_classes)))
                    if failure == "listing":
                        stack.enter_context(mock.patch.object(Path, "iterdir", autospec=True, side_effect=entries))
                    with self.assertRaises(PairingError) as caught:
                        main(argv)
                report = json.loads((output / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed_identity")
                self.assertEqual(report["error_type"], "PairingError")
                self.assertEqual(report["pairing_identity"], caught.exception.report)
                self.assertEqual(report["pairing_identity"]["component"], component)
                self.assertEqual(Path(report["pairing_identity"]["path"]), failed_path.resolve())
                if error_type is not None:
                    self.assertEqual(report["pairing_identity"]["error_type"], error_type)
                for native_class in native_classes:
                    native_class.from_pretrained.assert_not_called()
                runtime.probe_relevance.assert_not_called()
                runtime.Image.open.assert_not_called()
                self.assertFalse((output / "results.jsonl").exists())
