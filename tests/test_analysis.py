import unittest
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd

from analysis.between_essay_stability_analyzer import BetweenEssayStabilityAnalyzer
from analysis.bootstrap_confidence_interval import BootstrapConfidenceInterval
from analysis.descriptive_statistics import DescriptiveStatistics
from analysis.hotelling_t2_test import HotellingT2Test
from analysis.olms_vector_table import OLMSVectorTable
from analysis.prompt_shift_analyzer import PromptShiftAnalyzer
from analysis.response_level_bootstrapper import ResponseLevelBootstrapper
from analysis.vector_centroid_transformer import VectorCentroidTransformer
from analysis.vector_distance_transformer import VectorDistanceTransformer
from analysis.within_essay_stability_analyzer import WithinEssayStabilityAnalyzer


def vector_fixture(essay_count=2, count=3):
    rows = []
    for essay in range(essay_count):
        for comparison in ["AA", "BB", "AB"]:
            ids = range(1, count + 1)
            pairs = product(ids, ids) if comparison == "AB" else combinations_with_replacement(ids, 2)
            for i, j in pairs:
                # Self scores deliberately differ from one to catch bootstrap shortcuts.
                offset = {"AA": 0.2, "BB": 0.3, "AB": 0.1}[comparison]
                value = offset + essay * 0.01 + (i + j) * 0.02
                vector = np.array([value, value * 0.8, value + 0.1, value * 0.7])
                rows.append({"essay_file": f"essay_{essay}.md", "prompt_1": comparison[0],
                             "prompt_2": comparison[1], "comparison_type": comparison,
                             "response_index_1": i, "response_index_2": j, "response_count": count,
                             **dict(zip(OLMSVectorTable.COMPONENTS, vector))})
    return pd.DataFrame(rows)


class AnalysisTests(unittest.TestCase):
    def test_centroid_and_distance(self):
        vectors = np.array([[0, 0, 0, 0], [2, 4, 6, 8]])
        centroid = VectorCentroidTransformer.transform(vectors)
        np.testing.assert_array_equal(centroid, [1, 2, 3, 4])
        np.testing.assert_allclose(VectorDistanceTransformer.transform(vectors, centroid), np.sqrt(30))

    def test_statistics_and_undefined_cases(self):
        stats = DescriptiveStatistics.summarize([1, 2, 3, 4])
        self.assertEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["sd"], np.sqrt(5 / 3))
        self.assertEqual(stats["iqr"], 1.5)
        self.assertTrue(np.isnan(DescriptiveStatistics.summarize([0, 0])["cv"]))
        self.assertTrue(np.isnan(DescriptiveStatistics.summarize([1])["sd"]))

    def test_coverage_and_self_pairs(self):
        table = OLMSVectorTable(vector_fixture(count=10))
        for comparison, expected in [("AA", 45), ("BB", 45), ("AB", 100)]:
            self.assertEqual(len(table.select("essay_0.md", comparison)), expected)
        self.assertEqual(len(table.frame), 2 * 210)
        for mutate in [lambda frame: frame.iloc[:-1],
                       lambda frame: pd.concat([frame, frame.iloc[:1]]),
                       lambda frame: frame.assign(org=np.nan),
                       lambda frame: frame.assign(response_count=4),
                       lambda frame: frame.assign(comparison_type="BA")]:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                OLMSVectorTable(mutate(vector_fixture()))

    def test_within_estimates_exclude_diagonals(self):
        frame = vector_fixture(essay_count=1)
        self_rows = (frame.prompt_1 == frame.prompt_2) & (frame.response_index_1 == frame.response_index_2)
        frame.loc[self_rows, OLMSVectorTable.COMPONENTS] = 100
        table = OLMSVectorTable(frame)
        estimates, distances = WithinEssayStabilityAnalyzer().analyze(table)
        self.assertEqual(len(distances), 6)
        self.assertTrue((estimates.pair_count == 3).all())
        vectors = table.select("essay_0.md", "AA")[table.COMPONENTS].to_numpy()
        expected = np.sqrt(((vectors - vectors.mean(axis=0)) ** 2).sum(axis=1)).mean()
        self.assertAlmostEqual(estimates.iloc[0].mean_distance, expected)
        self.assertLess(estimates.iloc[0].org_mean, 1)

    def test_between_essay_uses_essay_estimates(self):
        within, _ = WithinEssayStabilityAnalyzer().analyze(OLMSVectorTable(vector_fixture(essay_count=3)))
        summary = BetweenEssayStabilityAnalyzer().analyze(within)
        self.assertEqual(len(summary), 10)
        self.assertTrue((summary["count"] == 3).all())
        row = summary[(summary.prompt == "A") & (summary.metric == "org_mean")].iloc[0]
        self.assertAlmostEqual(row["mean"], within[within.prompt == "A"].org_mean.mean())

    def test_shift_sign_and_mean_magnitude(self):
        frame = vector_fixture(essay_count=2)
        for essay, shift in [("essay_0.md", 0.2), ("essay_1.md", -0.2)]:
            frame.loc[frame.essay_file == essay, OLMSVectorTable.COMPONENTS] = 0.5
            frame.loc[(frame.essay_file == essay) & (frame.comparison_type == "AB"), "org"] += shift
        shifts, overall = PromptShiftAnalyzer().analyze(OLMSVectorTable(frame))
        np.testing.assert_allclose(shifts.delta_org, [0.2, -0.2])
        self.assertAlmostEqual(overall.iloc[0].magnitude, 0.2)
        self.assertAlmostEqual(overall.iloc[0].delta_org, 0)

    def test_bootstrap_matches_response_resampling_by_hand(self):
        table = OLMSVectorTable(vector_fixture(essay_count=1, count=3))
        actual = ResponseLevelBootstrapper(replicates=8, seed=17).run(table)
        rng = np.random.default_rng(17)
        a, b = rng.integers(0, 3, (8, 3)), rng.integers(0, 3, (8, 3))
        matrices = table.matrices("essay_0.md")
        self.assertTrue(any(len(set(sample)) < 3 for sample in a))
        for replicate in range(8):
            centroids = []
            for prompt, sample in [("A", a[replicate]), ("B", b[replicate])]:
                pairs = np.array([matrices[prompt * 2][sample[i], sample[j]]
                                  for i in range(3) for j in range(i + 1, 3)])
                centroid = pairs.mean(axis=0)
                centroids.append(centroid)
                distance = np.sqrt(((pairs - centroid) ** 2).sum(axis=1)).mean()
                row = actual["within_bootstrap"].query("prompt == @prompt and replicate == @replicate + 1").iloc[0]
                self.assertAlmostEqual(row.mean_distance, distance)
                np.testing.assert_allclose(row[[f"{c}_mean" for c in table.COMPONENTS]].to_numpy(dtype=float), centroid)
            ab = np.array([matrices["AB"][i, j] for i in a[replicate] for j in b[replicate]])
            delta = ab.mean(axis=0) - (centroids[0] + centroids[1]) / 2
            row = actual["prompt_shift_bootstrap"].iloc[replicate]
            np.testing.assert_allclose(row[[f"delta_{c}" for c in table.COMPONENTS]].to_numpy(dtype=float), delta)
            self.assertAlmostEqual(row.magnitude, np.linalg.norm(delta))

    def test_bootstrap_reproducibility_and_overall_aggregation(self):
        table = OLMSVectorTable(vector_fixture())
        first = ResponseLevelBootstrapper(251, 9).run(table)
        second = ResponseLevelBootstrapper(251, 9).run(table)
        for key in first:
            pd.testing.assert_frame_equal(first[key], second[key])
        expected = first["prompt_shift_bootstrap"].groupby("replicate").magnitude.mean()
        np.testing.assert_allclose(first["overall_prompt_shift_bootstrap"].magnitude, expected)
        with self.assertRaises(ValueError):
            ResponseLevelBootstrapper(5, -1)

    def test_intervals_and_directional_fraction(self):
        result = BootstrapConfidenceInterval(0.5).summarize([-2, -1, 0, 1, 2], directional=True)
        self.assertEqual((result["lower"], result["upper"]), (-1, 1))
        self.assertEqual(result["directional_p_like"], 0.6)
        with self.assertRaises(ValueError):
            BootstrapConfidenceInterval(1)

    def test_hotelling_known_covariance(self):
        mean = np.array([0.1, 0.2, 0.3, 0.4])
        shifts = np.concatenate([np.eye(4), -np.eye(4)]) + mean
        result = HotellingT2Test().run(shifts)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["t2"], 28 * (mean @ mean))
        self.assertAlmostEqual(result["f"], 4 * (mean @ mean))
        self.assertEqual((result["df1"], result["df2"]), (4, 4))
        self.assertTrue(0 < result["p_value"] < 1)
        self.assertEqual(HotellingT2Test().run(shifts[:4])["status"], "unavailable")
        self.assertEqual(HotellingT2Test().run(np.ones((8, 4)))["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
