"""Written for later execution; this implementation session does not run tests."""
import math
import random
import unittest
from dataclasses import replace
import torch
from v2_pruning.config import PruningConfig
from v2_pruning.coverage import build_utilities
from v2_pruning.regions import build_regions
from v2_pruning.selector import prune_visual_tokens, select_tokens
from v2_pruning.types import RegionMap


def config(**changes):
    return replace(PruningConfig(0.8, 0.5, 0.3, 0.75, 0.9), **changes)


def toy(axes):
    features = torch.eye(5, dtype=torch.float64)[torch.tensor(axes) - 1]
    relevance = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1], dtype=torch.float64)
    return prune_visual_tokens(features, relevance, (2, 4), config())


def independent_h(a, w, regions, selected, cfg):
    """Scalar definition independent of incremental state and tensor reductions."""
    n = len(w)
    constraints = [list(range(n))] + regions
    targets = [cfg.alpha_all] + [cfg.alpha_local] * len(regions)
    score, rates = 0.0, []
    for number, (demands, target) in enumerate(zip(constraints, targets)):
        representatives = selected if number == 0 else [j for j in selected if j in demands]
        denominator = math.fsum(w[i] * a[i][i] for i in demands)
        numerator = math.fsum(w[i] * max([a[i][j] for j in representatives] or [0.0]) for i in demands)
        rate = numerator / denominator
        rates.append(rate)
        score += min(rate / target, 1.0)
    return score, rates


class SelectionTests(unittest.TestCase):
    def test_float32_target_underflow_cannot_report_empty_success(self):
        mapping = RegionMap(torch.tensor([1, 1]), [[0, 1]], [0], (1, 2))
        cfg = config(alpha_all=1e-46, alpha_local=1e-46, coverage_tolerance=0)
        result = select_tokens(torch.eye(2), torch.ones(2), mapping, cfg)
        self.assertEqual(result.status, "fallback_full")
        self.assertEqual(result.indices, [0, 1])

    def test_large_tolerance_cannot_turn_low_coverage_into_success(self):
        with self.assertRaises(ValueError):
            config(alpha_all=0.9, alpha_local=0.9, coverage_tolerance=0.8).validate()

    def test_documented_unit_vector_cases_have_dynamic_k(self):
        first = toy([1, 1, 4, 2, 1, 1, 4, 3])
        second = toy([1, 1, 4, 2, 5, 5, 4, 3])
        self.assertEqual(first.selection.order, [0, 3, 7])
        self.assertEqual(second.selection.order, [0, 4, 3, 7])
        for result in (first, second):
            self.assertEqual(result.selection.status, "satisfied")
            self.assertEqual(result.regions.regions, [[0, 1, 4, 5], [3, 7]])
            self.assertEqual(result.selection.indices, sorted(result.selection.order))
            self.assertEqual(len(result.selection.trace), len(result.selection.order) + 1)
            self.assertIsNone(result.selection.trace[0].selected_index)

    def test_every_random_greedy_step_matches_independent_exhaustive_h(self):
        rng = random.Random(17)
        for n in range(4, 9):
            matrix = [[rng.random() * 0.8 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                matrix[i][i] = 1.0
            weights = [1 + rng.random() for _ in range(n)]
            regions = [list(range(n // 2)), list(range(n // 2, n - 1))]
            labels = [1 if i in regions[0] else 2 if i in regions[1] else 0 for i in range(n)]
            mapping = RegionMap(torch.tensor(labels), regions, [], (1, n))
            cfg = config(alpha_all=0.94, alpha_local=0.97, gain_chunk_size=2, coverage_tolerance=0)
            result = select_tokens(torch.tensor(matrix, dtype=torch.float64),
                                   torch.tensor(weights, dtype=torch.float64), mapping, cfg)
            self.assertEqual(result.status, "satisfied")
            selected = []
            for step in result.trace[1:]:
                before, _ = independent_h(matrix, weights, regions, selected, cfg)
                gains = {k: independent_h(matrix, weights, regions, selected + [k], cfg)[0] - before
                         for k in range(n) if k not in selected}
                best = min(gains, key=lambda k: (-gains[k], k))
                self.assertEqual(step.selected_index, best)
                self.assertAlmostEqual(step.gain, gains[best], places=11)
                selected.append(best)
                score, rates = independent_h(matrix, weights, regions, selected, cfg)
                self.assertAlmostEqual(step.progress, score, places=11)
                for actual, expected in zip([step.global_coverage] + step.local_coverages, rates):
                    self.assertAlmostEqual(actual, expected, places=11)

    def test_cross_region_similarity_does_not_supply_local_credit(self):
        mapping = RegionMap(torch.tensor([1, 2]), [[0], [1]], [0, 1], (1, 2))
        result = select_tokens(torch.ones(2, 2), torch.ones(2), mapping, config(max_tokens=1))
        self.assertEqual(result.order, [0])
        self.assertEqual(result.trace[-1].global_coverage, 1)
        self.assertEqual(result.trace[-1].local_coverages, [1, 0])
        self.assertEqual(result.status, "resource_limited_unsatisfied")

    def test_background_remains_selectable_without_budget_backfill(self):
        mapping = RegionMap(torch.tensor([1, 0, 0]), [[0]], [0], (1, 3))
        result = select_tokens(torch.eye(3), torch.ones(3), mapping,
                               config(alpha_all=0.6, max_tokens=3))
        self.assertEqual(result.order, [0, 1])
        self.assertEqual(result.status, "satisfied")
        self.assertLess(len(result.indices), 3)

    def test_ties_across_chunks_choose_smallest_unselected_index(self):
        mapping = RegionMap(torch.ones(5, dtype=torch.long), [list(range(5))], [0], (1, 5))
        for chunk in (1, 2, 8):
            result = select_tokens(torch.eye(5), torch.ones(5), mapping,
                                   config(alpha_all=1, alpha_local=1, gain_chunk_size=chunk))
            self.assertEqual(result.order, list(range(5)))

    def test_numerical_failure_returns_all_and_keeps_attempt_history(self):
        mapping = RegionMap(torch.tensor([1, 1]), [[0, 1]], [0], (1, 2))
        result = select_tokens(torch.eye(2), torch.ones(2), mapping,
                               config(alpha_all=1, alpha_local=1, gain_epsilon=0.1))
        self.assertEqual(result.status, "satisfied")
        blocked = select_tokens(torch.eye(2), torch.tensor([10., 1.]), mapping,
                                config(alpha_all=1, alpha_local=1, gain_epsilon=0.2))
        self.assertEqual(blocked.status, "fallback_full")
        self.assertEqual(blocked.indices, [0, 1])
        self.assertEqual(blocked.order, [0])
        self.assertEqual(len(blocked.trace), 2)
        self.assertLess(blocked.trace[-1].global_coverage, 1)
        bad_local = RegionMap(torch.tensor([0, 1]), [[1]], [1], (1, 2))
        bad = select_tokens(torch.diag(torch.tensor([1., 0.])), torch.ones(2), bad_local, config())
        self.assertEqual(bad.status, "fallback_full")
        self.assertIn("denominator", bad.reason)

    def test_sample_degeneracies_and_invalid_spatial_map_retain_all(self):
        good = torch.eye(3)
        raw = torch.tensor([0., 0.5, 1.])
        cases = [(good, torch.ones(3), (1, 3)), (good, raw * float("nan"), (1, 3)),
                 (good * 0, raw, (1, 3)), (torch.ones(3, 2), raw, (1, 3)),
                 (good, raw, (2, 2)), (good, raw, (True, 3))]
        for features, relevance, shape in cases:
            with self.subTest(shape=shape):
                result = prune_visual_tokens(features, relevance, shape, config())
                self.assertEqual(result.selection.status, "fallback_full")
                self.assertEqual(result.selection.indices, [0, 1, 2])
        overflow = prune_visual_tokens(good, raw, (1, 3), config(tau=1000))
        self.assertEqual(overflow.selection.status, "fallback_full")

    def test_raw_small_nonzero_vectors_use_float32_and_double_is_preserved(self):
        tiny = torch.eye(2, dtype=torch.float16) * 1e-7
        self.assertEqual(build_utilities(tiny, config()).dtype, torch.float32)
        self.assertEqual(build_utilities(torch.eye(2, dtype=torch.float64), config()).dtype, torch.float64)

    def test_utilities_are_row_centered_on_full_set_and_can_be_asymmetric(self):
        features = torch.tensor([[1., 0.], [1., 1.], [0., 1.]], dtype=torch.float64)
        utilities = build_utilities(features, config())
        unit = features / features.norm(dim=1, keepdim=True)
        cosine = unit @ unit.T
        expected = (cosine - cosine.mean(dim=1, keepdim=True)).clamp_min(0)
        torch.testing.assert_close(utilities, expected)
        self.assertNotEqual(float(utilities[0, 1]), float(utilities[1, 0]))

    def test_regions_use_four_neighbours_and_single_peak_plateau(self):
        diagonal = build_regions(torch.tensor([1., 0., 0., 0.9]), (2, 2), config())
        self.assertEqual(diagonal.regions, [[0], [3]])
        plateau = build_regions(torch.tensor([1., 1., 0.9, 0.9, 0., 0.]), (2, 3), config())
        self.assertEqual(plateau.seeds, [0])
        self.assertEqual(plateau.regions, [[0, 1, 2, 3]])

    def test_resource_limit_zero_is_not_satisfied_or_fallback(self):
        mapping = RegionMap(torch.tensor([1, 1]), [[0, 1]], [], (1, 2))
        result = select_tokens(torch.eye(2), torch.ones(2), mapping, config(max_tokens=0))
        self.assertEqual(result.status, "resource_limited_unsatisfied")
        self.assertEqual(result.indices, [])

    def test_empty_visual_sequence_has_explicit_status(self):
        result = prune_visual_tokens(torch.empty(0, 3), torch.empty(0), (0, 0), config())
        self.assertEqual(result.selection.status, "no_visual_tokens")
        self.assertEqual(result.selection.indices, [])

    def test_illegal_configuration_raises_instead_of_fallback(self):
        base = dict(seed_threshold=0.8, grow_threshold=0.5, tau=0.3, alpha_all=0.75, alpha_local=0.9)
        for change in ({"unknown": 1}, {"tau": float("nan")}, {"tau": -1},
                       {"alpha_all": 0}, {"alpha_local": 1.1}, {"grow_threshold": 0.9},
                       {"seed_threshold": True}, {"gain_chunk_size": 1.5}, {"max_tokens": True},
                       {"norm_epsilon": 0}, {"coverage_tolerance": 0.9}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                PruningConfig.from_dict({**base, **change})
        with self.assertRaises(ValueError):
            PruningConfig.from_dict({})
        with self.assertRaises(ValueError):
            prune_visual_tokens(torch.eye(2), torch.arange(2.), (1, 2), config(tau=-1))


if __name__ == "__main__":
    unittest.main()
