"""Offline adapter contracts, written but not executed in this delivery."""
import json
from pathlib import Path
import tempfile
import unittest

from v2_pruning.cli import fallback_reason, read_questions, resolve_image_path
from v2_pruning.llava_adapter import question_token_span, resolve_grid_shape, validate_loading_info


class AdapterContractTests(unittest.TestCase):
    def test_empty_and_whitespace_questions_have_no_probe_queries(self):
        # Tokenizer prefix alignment alone could include a boundary/template token.
        for question in ("", "   ", "\n\t"):
            with self.subTest(question=question):
                span = question_token_span([1, -200, 7], [1, -200, 6],
                                           [1, -200, 6], question)
                self.assertEqual(list(span), [])

    def test_reliable_grid_requires_matching_patch_dimensions_and_token_count(self):
        self.assertEqual(resolve_grid_shape(336, 336, 14, 576), ((24, 24), None))
        for height, width, patch, count in ((337, 336, 14, 576), (336, 336, 14, 575),
                                           (336, 336, 0, 576), (0, 336, 14, 576)):
            with self.subTest(height=height, width=width, patch=patch, count=count):
                grid, reason = resolve_grid_shape(height, width, patch, count)
                self.assertIsNone(grid)
                self.assertIsInstance(reason, str)
                self.assertTrue(reason)

    def test_missing_question_or_grid_records_fallback_reason(self):
        from types import SimpleNamespace
        valid = SimpleNamespace(grid_shape=(24, 24), grid_error=None)
        invalid = SimpleNamespace(grid_shape=None, grid_error="visual rows and grid disagree")
        self.assertIn("no effective question", fallback_reason(valid, "  "))
        self.assertIn("grid disagree", fallback_reason(invalid, "count?"))
        self.assertIsNone(fallback_reason(valid, "count?"))

    def test_empty_question_is_valid_input_to_the_full_keep_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").write_bytes(b"fixture")
            path = root / "questions.jsonl"
            path.write_text(json.dumps({"question_id": 1, "image": "image.jpg", "question": "  "}) + "\n", encoding="utf-8")
            self.assertEqual(read_questions(path, root)[0]["question"], "  ")

    def test_question_span_does_not_assume_bpe_concatenation(self):
        # Token 7 crosses the question's left character boundary; token 9 its right.
        full = [1, -200, 7, 8, 9, 12]
        self.assertEqual(list(question_token_span(full, [1, -200, 6], [1, -200, 7, 8, 10])), [2, 3, 4])

    def test_question_span_keeps_exact_prefix_boundaries(self):
        self.assertEqual(list(question_token_span([1, -200, 7, 8, 9], [1, -200], [1, -200, 7, 8])), [2, 3])

    def test_missing_projector_or_decoder_weight_is_rejected(self):
        for key in ("model.mm_projector.0.weight", "model.layers.0.self_attn.q_proj.weight",
                    "vision_model.encoder.layers.0.self_attn.q_proj.weight"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_loading_info({"missing_keys": [key]}, "test")

    def test_only_declared_unexpected_keys_are_allowed(self):
        validate_loading_info({"unexpected_keys": ["text_model.layer.weight"]}, "vision", ("text_model.",))
        with self.assertRaises(ValueError):
            validate_loading_info({"unexpected_keys": ["vision_model.wrong.weight"]}, "vision", ("text_model.",))

    def test_upstream_question_fields_and_reference_remain_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").write_bytes(b"fixture, not decoded by this reader")
            data = {"question_id": 2, "image": "image.jpg", "text": "How many?", "answers": ["2"]}
            (root / "questions.jsonl").write_text(json.dumps(data) + "\n", encoding="utf-8")
            rows = read_questions(root / "questions.jsonl", root)
            self.assertEqual(rows[0]["question"], "How many?")
            self.assertEqual(rows[0]["reference_answer"], ["2"])

    def test_question_image_cannot_escape_the_selected_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "outside.jpg").write_bytes(b"fixture")
            data = {"image": "../outside.jpg", "question": "Count?"}
            (root / "questions.jsonl").write_text(json.dumps(data) + "\n", encoding="utf-8")
            rows = read_questions(root / "questions.jsonl", root / "images")
            with self.assertRaises(ValueError):
                resolve_image_path(Path(rows[0]["image_path"]), root / "images")

    def test_nonstandard_json_and_invalid_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").write_bytes(b"fixture")
            path = root / "questions.jsonl"
            for identifier in (None, True, [], {}, "", float("nan")):
                with self.subTest(identifier=identifier):
                    path.write_text(json.dumps({"question_id": identifier, "image": "image.jpg",
                                                "question": "Count?"}) + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_questions(path, root)


if __name__ == "__main__":
    unittest.main()
