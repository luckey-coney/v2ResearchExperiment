"""Static report contracts requiring no model, network, or GPU resources."""
import base64
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from v2_pruning.reporting import write_case


SVG = {"svg": "http://www.w3.org/2000/svg"}


class ReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name) / "case"
        self.canvas = Image.new("RGB", (60, 40), (120, 160, 200))
        self.record = {
            "question": "图中有多少个对象？",
            "answer": "两个",
            "selection": {"status": "ok", "reason": None,
                          "indices": [0, 5], "order": [5, 0]},
            "visual_tokens_original": 6,
            "visual_tokens_kept": 2,
            "timing_ms": {"generation": 12.5},
        }

    def write_report(self, *, record=None, q=None, regions=None, selected=None):
        return write_case(
            self.output_dir, self.record if record is None else record,
            self.canvas, q, regions, [0, 5] if selected is None else selected,
            (2, 3),
        )

    def test_fixed_filenames_ignore_question_id(self):
        self.record["question_id"] = "../../outside/unsafe"
        paths = self.write_report()
        self.assertEqual(set(paths), {"json", "svg"})
        self.assertEqual(Path(paths["json"]), (self.output_dir / "result.json").resolve())
        self.assertEqual(Path(paths["svg"]), (self.output_dir / "visual_tokens.svg").resolve())
        self.assertEqual({p.name for p in self.output_dir.iterdir()},
                         {"result.json", "visual_tokens.svg"})

    def test_untrusted_text_is_escaped_and_images_are_embedded(self):
        payload = '<script>alert("x")</script>& <image href="https://invalid.test/x"/>'
        self.record.update(question=payload, answer=payload, reference_answer=payload)
        self.record["selection"].update(status=payload, reason=payload)
        self.record["timing_ms"] = {payload: 1.25}
        paths = self.write_report(q=[0., 1., 2., 3., 4., 5.], regions=[0, 1, 1, 0, 2, 2])
        source = Path(paths["svg"]).read_text(encoding="utf-8")
        root = ET.fromstring(source)
        self.assertIn("&lt;script&gt;", source)
        self.assertNotIn("<script>", source)
        self.assertIn(payload, "".join(root.itertext()))
        images = root.findall(".//svg:image", SVG)
        self.assertEqual(len(images), 3)
        self.assertEqual(len(root.findall(".//svg:rect[@class='heat-cell']", SVG)), 6)
        self.assertTrue(all(node.attrib["href"].startswith("data:image/png;base64,")
                            for node in images))
        self.assertEqual(len({node.attrib["href"] for node in images}), 1)
        png = base64.b64decode(images[0].attrib["href"].split(",", 1)[1])
        with Image.open(BytesIO(png)) as embedded:
            self.assertEqual(embedded.size, self.canvas.size)
            self.assertEqual(embedded.convert("RGB").tobytes(), self.canvas.tobytes())
        self.assertFalse(root.findall(".//svg:a", SVG))
        self.assertFalse(root.findall(".//svg:script", SVG))

    def test_actual_selection_uses_original_zero_based_grid(self):
        self.record["selection"]["indices"] = [1]
        self.record["selection"]["order"] = [1, 2, 3]
        paths = self.write_report(selected=[5, 0])
        root = ET.fromstring(Path(paths["svg"]).read_text(encoding="utf-8"))
        panel = root.find("svg:g[@id='retained-tokens']", SVG)
        cells = panel.findall("svg:rect[@data-index]", SVG)
        self.assertEqual(len(cells), 6)
        self.assertEqual({int(cell.attrib["data-index"]) for cell in cells
                          if cell.attrib["data-selected"] == "true"}, {0, 5})
        final_cell = next(cell for cell in cells if cell.attrib["data-index"] == "5")
        self.assertEqual((final_cell.attrib["data-row"], final_cell.attrib["data-col"]),
                         ("1", "2"))
        self.assertEqual(final_cell.attrib["fill"], "none")
        dropped = next(cell for cell in cells if cell.attrib["data-index"] == "1")
        self.assertEqual(dropped.attrib["fill-opacity"], "0.84")
        self.assertIn("K/N=2/6", "".join(root.itertext()))
        result = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        self.assertEqual(result["selection"]["order"], [1, 2, 3])
        self.assertEqual(result["report_metadata"]["selected_indices"], [5, 0])
        self.assertTrue(result["report_metadata"]["warnings"])

    def test_out_of_range_or_duplicate_indices_do_not_write_files(self):
        for indices in ([-1], [6], [0, 0], [True], [1.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.write_report(selected=indices)
        self.assertFalse(self.output_dir.exists())

    def test_missing_maps_are_marked_unavailable(self):
        paths = self.write_report()
        source = Path(paths["svg"]).read_text(encoding="utf-8")
        result = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        self.assertIn("问题相关性不可用", source)
        self.assertIn("重点区域不可用", source)
        self.assertNotIn('class="heat-cell"', source)
        self.assertFalse(result["report_metadata"]["q_status"]["available"])
        self.assertFalse(result["report_metadata"]["region_status"]["available"])

    def test_unmapped_full_fallback_preserves_answer_and_does_not_invent_grid(self):
        record = {
            "question": "图中有什么？", "answer": "一只猫", "reference_answer": "猫",
            "selection": {"status": "fallback_full", "reason": "visual grid mismatch",
                          "indices": [0, 1, 2], "order": []},
            "visual_tokens_original": 3, "visual_tokens_kept": 3,
            "timing_ms": {"generation": 12.5},
        }
        paths = write_case(self.output_dir, record, self.canvas, None, None, [0, 1, 2], None)
        data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        source = Path(paths["svg"]).read_text(encoding="utf-8")
        root = ET.fromstring(source)
        self.assertEqual(data["answer"], "一只猫")
        self.assertEqual(data["selection"]["reason"], "visual grid mismatch")
        self.assertIsNone(data["report_metadata"]["grid_shape"])
        self.assertEqual(data["report_metadata"]["visual_tokens_original"], 3)
        self.assertEqual(data["report_metadata"]["visual_tokens_kept"], 3)
        self.assertEqual(data["report_metadata"]["selected_indices"], [0, 1, 2])
        self.assertFalse(data["report_metadata"]["q_status"]["available"])
        self.assertFalse(data["report_metadata"]["region_status"]["available"])
        self.assertIn("空间映射不可用，保留完整视觉序列", source)
        self.assertIn("K/N=3/3", source)
        self.assertEqual(len(root.findall(".//svg:image", SVG)), 1)
        self.assertFalse(root.findall(".//svg:rect[@data-index]", SVG))
        self.assertNotIn('id="question-relevance"', source)
        self.assertNotIn('id="focus-regions"', source)
        self.assertNotIn('id="retained-tokens"', source)

    def test_unmapped_output_rejects_maps_or_inconsistent_full_keep(self):
        record = {
            "selection": {"status": "fallback_full", "reason": "invalid grid", "indices": [0, 1]},
            "visual_tokens_original": 2, "visual_tokens_kept": 2,
        }
        for q, regions, selected, overrides in (
            ([0.2, 0.8], None, [0, 1], {}),
            (None, [1, 1], [0, 1], {}),
            (None, None, [0], {}),
            (None, None, [0, 1], {"visual_tokens_original": 3}),
            (None, None, [0, 1], {"visual_tokens_kept": 1}),
            (None, None, [0, 1], {"selection": {"status": "satisfied", "indices": [0, 1]}}),
        ):
            with self.subTest(q=q, regions=regions, selected=selected, overrides=overrides):
                with self.assertRaises(ValueError):
                    write_case(self.output_dir, {**record, **overrides}, self.canvas,
                               q, regions, selected, None)
                self.assertFalse(self.output_dir.exists())

    def test_mismatched_q_is_not_rendered_as_a_heatmap(self):
        paths = self.write_report(q=[0.2], regions=[0, 0, 0, 0, 0, 0])
        source = Path(paths["svg"]).read_text(encoding="utf-8")
        self.assertIn("长度", source)
        self.assertNotIn('class="heat-cell"', source)

    def test_nonfinite_values_become_null_with_unavailability_reasons(self):
        self.record["timing_ms"] = {"generation": float("nan"), "decode": float("inf")}
        self.record["extra"] = [{"score": -float("inf")}]
        paths = self.write_report(q=[0., 1., float("nan"), 3., 4., 5.])
        source = Path(paths["json"]).read_text(encoding="utf-8")

        def reject_constant(value):
            self.fail(f"nonstandard JSON constant: {value}")

        result = json.loads(source, parse_constant=reject_constant)
        self.assertNotIn("NaN", source)
        self.assertNotIn("Infinity", source)
        self.assertIsNone(result["timing_ms"]["generation"])
        self.assertIsNone(result["extra"][0]["score"])
        self.assertIsNone(result["report_metadata"]["q"][2])
        notes = result["report_metadata"]["nonfinite_values"]
        self.assertEqual(len(notes), 4)
        self.assertTrue(all("非有限值不可用" in note["reason"] for note in notes))
        self.assertFalse(result["report_metadata"]["q_status"]["available"])
        self.assertTrue(math.isnan(self.record["timing_ms"]["generation"]))

    def test_missing_or_unknown_timing_does_not_claim_zero(self):
        for timing in (None, {}, {"generation": "unknown"}):
            with self.subTest(timing=timing):
                self.record["timing_ms"] = timing
                paths = self.write_report()
                source = Path(paths["svg"]).read_text(encoding="utf-8")
                self.assertIn("未测", source)
                self.assertNotIn("=0 ms", source)

    def test_unsupported_record_value_is_rejected_without_stringification(self):
        self.record["unsupported"] = object()
        with self.assertRaises(TypeError):
            self.write_report()
        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
