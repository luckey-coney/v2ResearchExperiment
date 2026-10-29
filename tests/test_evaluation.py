"""Standard-library tests; running these is separate from writing them."""
import json
import math
import unittest

from v2_pruning.evaluation import score_answer, summarize_records


def record(k=2, n=4, status="satisfied", prediction="yes", reference="yes", **extra):
    return {"quality": score_answer(prediction, reference),
            "visual_tokens_kept": k, "visual_tokens_original": n,
            "selection": {"status": status}, "region_count": 1,
            "generated_tokens": 3, "timing_ms": {"end_to_end": 12}, **extra}


class ExactMatchTests(unittest.TestCase):
    def test_known_normalization_and_no_answer_extraction(self):
        self.assertEqual(score_answer("  ＹＥＳ\tStraße  ", "yes STRASSE")["score"], 1.0)
        self.assertEqual(score_answer("yes.", "yes")["score"], 0.0)
        self.assertEqual(score_answer("The answer is yes", "yes")["score"], 0.0)

    def test_reference_list_is_any_reference_match(self):
        quality = score_answer("2", ["two", 2, 2.0])
        self.assertEqual(quality["score"], 1.0)
        self.assertEqual(quality["reference_count"], 3)
        self.assertEqual(score_answer("0", 0)["score"], 1.0)
        self.assertEqual(score_answer("2", 2.0)["score"], 0.0)

    def test_unlabeled_invalid_and_nonfinite_are_unscored(self):
        for reference in (None, "  ", [], [""], {}, ["yes", {}], True, math.nan, math.inf):
            with self.subTest(reference=reference):
                quality = score_answer("yes", reference)
                self.assertIsNone(quality["score"])
                self.assertTrue(quality["reason"].startswith("exact_match:"))

    def test_disabled_and_unsupported_protocol(self):
        quality = score_answer("yes", "yes", "none")
        self.assertIsNone(quality["score"])
        self.assertEqual(quality["reference_count"], 1)
        self.assertIn("disabled", quality["reason"])
        with self.assertRaises(ValueError):
            score_answer("yes", "yes", "official_vqa")


class SummaryTests(unittest.TestCase):
    def test_failed_and_unlabeled_are_not_in_quality_denominator(self):
        records = [record(), record(prediction="no"), record(reference=None)]
        summary = summarize_records(records, [{"error": "failed"}], total=5)
        self.assertEqual(summary["counts"], {"total": 5, "completed": 3, "failed": 1, "unprocessed": 1})
        quality = summary["quality"]
        self.assertEqual(quality["score_denominator"], 2)
        self.assertEqual(quality["accuracy_scored_only"], 0.5)
        self.assertEqual(quality["without_reference"], 1)
        self.assertEqual(quality["unscored_completed"], 1)
        self.assertEqual(quality["coverage_scored_over_total"], 2 / 5)
        self.assertEqual(quality["coverage_scored_over_completed"], 2 / 3)

    def test_linear_quantiles(self):
        summary = summarize_records([record(k=k) for k in range(1, 5)], [], 4)
        stats = summary["visual_tokens_kept"]
        self.assertEqual(stats["count"], 4)
        for key, value in {"mean": 2.5, "median": 2.5, "p90": 3.7,
                           "p95": 3.85, "min": 1, "max": 4}.items():
            self.assertAlmostEqual(stats[key], value)
        self.assertIn("linear interpolation", summary["quantile_method"])

    def test_fallback_full_keeps_hard_samples_and_uses_actual_k(self):
        hard = record(k=8, n=8, status="fallback_full")
        hard["selection"]["trace"] = [{"global_coverage": 0.2}]
        summary = summarize_records([record(k=1, n=2), hard], [], 2)
        self.assertEqual(summary["visual_tokens_kept"]["mean"], 4.5)
        self.assertEqual(summary["compression"]["retained_fraction_weighted"], 0.9)
        self.assertAlmostEqual(summary["compression"]["reduction_fraction_weighted"], 0.1)
        self.assertEqual(summary["selection"]["fallback_full"], {"count": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(summary["selection"]["zero_compression"], {"count": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(summary["by_selection_status"]["fallback_full"]["visual_tokens_kept"]["max"], 8)
        self.assertNotIn("global_coverage", summary)

    def test_invalid_numbers_do_not_poison_stats_or_json(self):
        broken = record(k=math.nan, region_count=math.inf, generated_tokens=-1,
                        timing_ms={"end_to_end": math.nan, "probe": -1, "ttft": None})
        broken["quality"]["score"] = math.nan
        summary = summarize_records([record(), broken], [], 2)
        self.assertEqual(summary["visual_tokens_kept"]["count"], 1)
        self.assertEqual(summary["timing_ms"]["end_to_end"]["count"], 1)
        self.assertEqual(summary["timing_ms"]["probe"]["count"], 0)
        self.assertEqual(summary["region_count"]["count"], 1)
        self.assertEqual(summary["generated_tokens"]["count"], 1)
        self.assertEqual(summary["quality"]["score_denominator"], 1)
        json.dumps(summary, allow_nan=False)

    def test_none_empty_and_count_validation(self):
        summary = summarize_records([record()], [], 1, "none")
        self.assertIsNone(summary["quality"]["accuracy_scored_only"])
        self.assertEqual(summary["quality"]["score_denominator"], 0)
        empty = summarize_records([], [], 0)
        self.assertIsNone(empty["visual_tokens_kept"]["mean"])
        self.assertIsNone(empty["compression"]["retained_fraction_weighted"])
        with self.assertRaises(ValueError):
            summarize_records([record()], [], 0)

    def test_protocol_mismatch_is_not_silently_scored(self):
        item = record()
        item["quality"]["protocol"] = "external_unknown"
        summary = summarize_records([item], [], 1)
        self.assertEqual(summary["quality"]["score_denominator"], 0)
        self.assertEqual(summary["quality"]["unscored_reasons"], {"exact_match: protocol_mismatch": 1})

    def test_selection_rates_have_completed_denominator(self):
        summary = summarize_records([record(), record(k=4, status="resource_limited_unsatisfied")],
                                    [{"error": "failed"}], 3)
        self.assertEqual(summary["selection"]["satisfied"], {"count": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(summary["selection"]["resource_limited_unsatisfied"],
                         {"count": 1, "denominator": 2, "rate": 0.5})


if __name__ == "__main__":
    unittest.main()
