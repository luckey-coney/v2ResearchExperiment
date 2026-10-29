"""Startup failure/retry and input-audit regression contracts; not executed."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest import TestCase, mock

from v2_pruning.cli import main, input_audit


class CliLifecycleTests(TestCase):
    def test_missing_dependency_never_claims_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "config.json").write_text(json.dumps({"seed_threshold": .8, "grow_threshold": .5,
                 "tau": .3, "alpha_all": .75, "alpha_local": .9}), encoding="utf-8")
            (root / "questions.jsonl").write_text(json.dumps({"image": "missing.png", "question": "Count?"}) + "\n", encoding="utf-8")
            output = root / "output"
            argv = ["--model-path", str(root / "model"), "--vision-path", str(root / "vision"),
                    "--config", str(root / "config.json"), "--question-file", str(root / "questions.jsonl"),
                    "--image-folder", str(root / "images"), "--output-dir", str(output)]
            # Check both an early import and the real loader's lazy-backend preflight.
            for target, message in (("v2_pruning.cli._load_runtime", "missing torch"),
                                    ("v2_pruning.llava_adapter.preflight_runtime", "missing sentencepiece")):
                for _ in range(2):
                    with mock.patch(target, side_effect=ModuleNotFoundError(message)), \
                            self.assertRaisesRegex(RuntimeError, "output directory was not created"):
                        main(argv)
                    self.assertFalse(output.exists())

    def test_audit_distinguishes_original_and_compact_sequence_positions(self):
        class FakeTensor:
            def __init__(self, data):
                self.data = data
                self.shape = (len(data), len(data[0]))

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.data

        prepared = SimpleNamespace(prompt="full template", token_ids=FakeTensor([[1, 2, 1, 1, 1, 7, 8]]),
                                   position_ids=FakeTensor([list(range(7))]),
                                   question_positions=[5], visual_positions=[2, 3, 4])
        audit = input_audit(prepared, [1])
        self.assertEqual(audit["retained_original_visual_indices"], [1])
        self.assertEqual(audit["retained_original_sequence_indices"], [0, 1, 3, 5, 6])
        self.assertEqual(audit["generation_position_ids"], [[0, 1, 2, 3, 4]])
        self.assertEqual(audit["probe_position_ids"], [list(range(7))])
        self.assertEqual(audit["question_positions"], [5])
        self.assertIn("placeholder", audit["token_id_note"])
