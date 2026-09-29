"""Synthetic checks for the scientific estimands and incomplete-run detection."""

import csv
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location("pocman_k_analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixture():
    episodes, steps, probes = [], [], []
    for seed, length, latency in ((10, 1, 0.001), (20, 3, 0.009)):
        for k in (4, 16):
            episodes.append(dict(seed=seed, k=k, steps=length, **{"return": float(length)},
                                 discounted_return=float(length), terminal=0))
            metrics = dict(planning_wall_s=latency, planning_cpu_s=latency, search_cpu_s=latency / 2,
                           tree_nodes=length * 10, policy_nodes=length * 2, expanded_nodes=length,
                           trials=length, initial_gap=10.0, final_gap=2.0, belief_particles=100)
            for step in range(length):
                steps.append(dict(seed=seed, k=k, step=step, action=0, observation=0, reward=1.0,
                                  terminal=0, **metrics))
            for step in range(length):
                actions = (0, 0, 1) if seed == 10 else (1, 1, 1)
                for repeat, action in enumerate(actions):
                    probes.append(dict(seed=seed, step=step, k=k, repeat=repeat,
                                       planner_seed=seed * 100 + step * 10 + repeat, action=action, **metrics))
    return episodes, steps, probes


class AnalysisTests(unittest.TestCase):
    def test_pairwise_agreement_is_not_modal_share(self):
        result = analysis.action_statistics([0, 0, 1, 1])
        self.assertAlmostEqual(result["pairwise_agreement"], 1 / 3)
        self.assertEqual(result["modal_share"], 0.5)
        self.assertEqual(result["entropy_bits"], 1)
        self.assertEqual(result["modal_action"], 0)
        self.assertEqual(analysis.action_statistics([2, 2, 2])["pairwise_agreement"], 1)

    def test_planning_means_weight_episodes_equally(self):
        episodes, steps, probes = fixture()
        analysis.validate_inputs(episodes, steps, probes)
        summary = analysis.summarize(episodes, steps, analysis.stability_rows(probes), 500)
        self.assertEqual(summary[0]["mean_planning_wall_ms"], 5)
        self.assertEqual(summary[0]["mean_tree_nodes"], 20)
        self.assertEqual(summary[0]["mean_return"], 2)

    def test_seed_clusters_weight_seeds_equally_and_bootstrap_clusters(self):
        episodes, steps, probes = fixture()
        summary = analysis.summarize(episodes, steps, analysis.stability_rows(probes), 500)
        # Seed 10 contributes 1 belief at 1/3, seed 20 has 3 beliefs at 1.
        # Equal belief weighting would incorrectly give 5/6.
        self.assertAlmostEqual(summary[0]["mean_pairwise_agreement"], 2 / 3)
        self.assertAlmostEqual(summary[0]["pairwise_agreement_ci_low"], 1 / 3)
        self.assertEqual(summary[0]["pairwise_agreement_ci_high"], 1)
        doubled = probes + [dict(row, step=row["step"] + 100) for row in probes if row["seed"] == 20]
        summary_doubled = analysis.summarize(episodes, steps, analysis.stability_rows(doubled), 500)
        self.assertEqual(summary, [dict(row, probe_beliefs=4) for row in summary_doubled])

    def test_single_independent_seed_has_no_confidence_interval(self):
        self.assertEqual(analysis.mean_ci([0.5], 100, random.Random(1)), (None, None))
        episodes, steps, probes = fixture()
        episodes, steps, probes = [[row for row in rows if row["seed"] == 20]
                                   for rows in (episodes, steps, probes)]
        summary = analysis.summarize(episodes, steps, analysis.stability_rows(probes), 100)
        self.assertEqual(summary[0]["mean_pairwise_agreement"], 1)
        self.assertIsNone(summary[0]["pairwise_agreement_ci_low"])
        self.assertIsNone(summary[0]["return_ci_high"])

    def test_largest_k_has_no_self_benchmark_and_mode_ties_are_deterministic(self):
        _, _, probes = fixture()
        rows = analysis.stability_rows(probes)
        for row in rows:
            if row["k"] == 16:
                self.assertIsNone(row["agreement_to_max_k_mode"])
            elif row["seed"] == 10:
                self.assertAlmostEqual(row["agreement_to_max_k_mode"], 2 / 3)
        tiny = [dict(seed=1, step=0, k=k, action=action) for k in (4, 16) for action in (2, 1)]
        self.assertEqual(analysis.stability_rows(tiny)[0]["max_k_mode_action"], 1)

    def test_return_delta_bootstraps_matching_seed_pairs(self):
        episodes, steps, probes = fixture()
        for row in episodes:
            if row["k"] == 16:
                row["return"] += 2
        summary = analysis.summarize(episodes, steps, analysis.stability_rows(probes), 100)
        self.assertEqual(summary[0]["mean_paired_return_delta_vs_max_k"], -2)
        self.assertEqual(summary[0]["paired_return_delta_vs_max_k_ci_low"], -2)
        self.assertEqual(summary[0]["paired_return_delta_vs_max_k_ci_high"], -2)
        self.assertEqual(summary[1]["mean_paired_return_delta_vs_max_k"], 0)
        self.assertIsNone(summary[1]["paired_return_delta_vs_max_k_ci_low"])

    def test_truncated_or_duplicate_inputs_fail(self):
        episodes, steps, probes = fixture()
        with self.assertRaisesRegex(analysis.InputError, "consecutive steps"):
            analysis.validate_inputs(episodes, steps[:-1], probes)
        with self.assertRaisesRegex(analysis.InputError, "duplicate"):
            analysis.validate_inputs(episodes, steps, probes + [probes[0]])
        with self.assertRaisesRegex(analysis.InputError, "incomplete repeats"):
            analysis.validate_inputs(episodes, steps, probes[:-1])

    def test_csv_validation_and_end_to_end_without_matplotlib(self):
        episodes, steps, probes = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "settings": {"seeds": [10, 20], "ks": [4, 16]}}))
            for name, rows in (("episodes", episodes), ("steps", steps), ("probes", probes)):
                analysis.write_csv(root / f"{name}.csv", rows)
            with mock.patch.object(analysis, "make_plots", return_value=False):
                summary = analysis.analyze(root, 100)
            self.assertEqual(len(summary), 2)
            self.assertIn("finite benchmark", (root / "report.md").read_text())
            with (root / "stability.csv").open() as handle:
                largest = [row for row in csv.DictReader(handle) if row["k"] == "16"]
            self.assertTrue(all(row["agreement_to_max_k_mode"] == "" for row in largest))
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "settings": {"ks": [4, 16, 64]}}))
            with self.assertRaisesRegex(analysis.InputError, "disagree with manifest"):
                analysis.load_inputs(root)
            episodes[0]["return"] = float("nan")
            analysis.write_csv(root / "episodes.csv", episodes)
            with self.assertRaisesRegex(analysis.InputError, "return must be finite"):
                analysis.load_inputs(root)


if __name__ == "__main__":
    unittest.main()
