"""Regression contracts for the CLI fallback path; written, not executed.

These fixtures mock model execution. They do not establish that a real VLM or
GPU works, nor do they replace the full-retention equivalence check.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import json
import tempfile

from PIL import Image
import torch

from v2_pruning.cli import main


class FallbackFlowTests(TestCase):
    def _check_case(self, question, grid, grid_error, missing_image_first=False,
                    decompression_bomb_first=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (4, 4), "white").save(images / "image.png")
            questions = root / "questions.jsonl"
            first_failure = missing_image_first or decompression_bomb_first
            first_image = "missing.png" if missing_image_first else "image.png"
            first = json.dumps({"question_id": 0, "image": first_image, "question": question}) + "\n"
            questions.write_text((first if first_failure else "") +
                                 json.dumps({"question_id": 1, "image": "image.png",
                                             "question": question}) + "\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({"seed_threshold": 0.8, "grow_threshold": 0.5,
                                          "tau": 0.3, "alpha_all": 0.75,
                                          "alpha_local": 0.9}), encoding="utf-8")
            output = root / "output"
            calls = []
            open_calls = []
            original_open = Image.open

            def open_image(*args, **kwargs):
                if decompression_bomb_first and not open_calls:
                    open_calls.append("blocked")
                    raise Image.DecompressionBombError("fixture image exceeds Pillow pixel limit")
                open_calls.append("opened")
                return original_open(*args, **kwargs)

            class FakeAdapter:
                def __init__(self, *args, **kwargs):
                    self.device = "cuda:0"
                    self.model = object()
                    self.loading_info = {}
                    self.pairing_info = {"status": "mock_fixture_only"}

                def synchronize(self):
                    pass

                def prepare(self, image, user_question, conv_mode):
                    calls.append(("prepare", user_question))
                    return SimpleNamespace(
                        grid_shape=grid, grid_error=grid_error, visual_positions=[2, 3, 4, 5],
                        question_positions=[], inputs_embeds=None,
                        position_ids=torch.arange(8).unsqueeze(0), token_ids=torch.ones(1, 8, dtype=torch.long),
                        v0=None, original_size=image.size, prompt="fixture prompt", stage_timings_ms={},
                    )

                def generate(self, prepared, indices, max_new_tokens):
                    calls.append(("generate", indices))
                    return {"answer": "fixture response", "generated_tokens": 1,
                            "generation_ms": 1.0, "generation_ttft_ms": 1.0,
                            "input_sequence_length": 8, "position_policy": "compact"}

                def canvas(self, prepared):
                    return Image.new("RGB", (4, 4), "white")

            argv = ["--model-path", str(root / "model"), "--vision-path", str(root / "vision"),
                    "--config", str(config), "--question-file", str(questions),
                    "--image-folder", str(images), "--output-dir", str(output)]
            with mock.patch("v2_pruning.llava_adapter.LlavaAdapter", FakeAdapter), \
                    mock.patch("v2_pruning.llava_adapter.preflight_runtime"), \
                    mock.patch("v2_pruning.probe.probe_relevance") as probe, \
                    mock.patch("v2_pruning.selector.prune_visual_tokens") as selector, \
                    mock.patch("PIL.Image.open", side_effect=open_image), \
                    mock.patch("torch.cuda.manual_seed_all"), \
                    mock.patch("torch.cuda.get_device_name", return_value="fixture GPU"):
                if first_failure:
                    with self.assertRaises(SystemExit) as exit_context:
                        main(argv)
                    self.assertEqual(exit_context.exception.code, 1)
                else:
                    main(argv)
            probe.assert_not_called()
            selector.assert_not_called()
            self.assertIn(("generate", [0, 1, 2, 3]), calls)
            case = output / ("case_000002" if first_failure else "case_000001")
            result = json.loads((case / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["selection"]["status"], "fallback_full")
            self.assertEqual(result["visual_tokens_kept"], result["visual_tokens_original"])
            self.assertEqual(result["answer"], "fixture response")
            self.assertEqual(result["input_audit"]["visual_positions"], [2, 3, 4, 5])
            self.assertEqual(result["input_audit"]["retained_original_sequence_indices"], list(range(8)))
            self.assertEqual(result["input_audit"]["generation_position_ids"], [list(range(8))])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"], {"total": 2 if first_failure else 1,
                                               "completed": 1, "failed": int(first_failure),
                                               "unprocessed": 0})
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "finished_with_errors" if first_failure else "finished")
            for key in ("total", "completed", "failed"):
                self.assertEqual(run[key], summary["counts"][key])
            self.assertEqual(run["selection_status_counts"], {"fallback_full": 1})
            for filename in ("results.jsonl", "answers_agile_format.jsonl"):
                rows = [json.loads(line) for line in (output / filename).read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["question_id"] for row in rows], [1])
            self.assertEqual(summary["quality"]["score_denominator"], 0)
            self.assertEqual(summary["selection"]["fallback_full"]["rate"], 1.0)
            self.assertEqual(summary["compression"]["reduction_fraction_weighted"], 0.0)
            self.assertEqual(result["quality"]["protocol"], "exact_match")
            self.assertIsNone(result["quality"]["score"])
            self.assertEqual(result["input_audit"]["prompt"], "fixture prompt")
            errors = [json.loads(line) for line in (output / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([error["question_id"] for error in errors], [0] if first_failure else [])
            if missing_image_first:
                self.assertIn(errors[0]["error_type"], ("FileNotFoundError", "OSError"))
            if decompression_bomb_first:
                self.assertEqual(open_calls, ["blocked", "opened"])
                self.assertEqual(errors[0]["error_type"], "DecompressionBombError")
                self.assertEqual(errors[0]["error"], "fixture image exceeds Pillow pixel limit")
                self.assertEqual(errors[0]["image"], "image.png")
                self.assertFalse((output / "case_000001").exists())
            svg = (case / "visual_tokens.svg").read_text(encoding="utf-8")
            if grid is None:
                self.assertIsNone(result["grid_shape"])
                self.assertEqual(result["spatial_mapping_error"], grid_error)
                self.assertIn("空间映射不可用，保留完整视觉序列", svg)
                self.assertNotIn('data-index="', svg)
            else:
                self.assertEqual(result["grid_shape"], list(grid))
                self.assertIn("no effective question", result["selection"]["reason"])

    def test_empty_question_retains_all_visual_tokens(self):
        self._check_case("  ", (2, 2), None)

    def test_unmapped_grid_retains_all_visual_tokens(self):
        self._check_case("What is visible?", None, "visual rows and grid disagree")

    def test_missing_image_does_not_prevent_later_valid_sample(self):
        self._check_case("  ", (2, 2), None, missing_image_first=True)

    def test_decompression_bomb_does_not_prevent_later_valid_sample(self):
        self._check_case("  ", (2, 2), None, decompression_bomb_first=True)
