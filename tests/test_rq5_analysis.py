"""Unit tests for RQ5 feasibility check and leak-free recommendation label prediction."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_rq5 as rq5


class RQ5AnalysisTests(unittest.TestCase):
    def _synthetic_data(self):
        rng = np.random.default_rng(42)
        n_authors = 60
        rows = []
        weights_rows = []
        for idx in range(n_authors):
            # 50 single-game authors, 10 two-game authors
            n_games = 2 if idx < 10 else 1
            for g in range(n_games):
                rid = f"r_{idx}_{g}"
                appid = (idx + g) % 5 + 1
                lang = "english" if (idx + g) % 2 == 0 else "schinese"
                t03 = float(rng.uniform(0.0, 1.0))
                t05 = float(rng.uniform(0.0, 1.0 - t03))
                t01 = 1.0 - t03 - t05
                # Higher T03 -> lower rec probability, higher T05 -> higher rec probability
                logit = 0.8 - 4.0 * t03 + 3.0 * t05
                rec = int(rng.random() < (1.0 / (1.0 + np.exp(-logit))))
                rows.append({
                    "review_id": rid,
                    "steamid": f"u_{idx}",
                    "appid": appid,
                    "appid_str": str(appid),
                    "language": lang,
                    "recommend": rec,
                    "timestamp_created": 1_700_000_000 + idx * 1000 + g * 50000,
                    "association_eligible": True,
                    "playtime_analysis_eligible": True,
                    "hours": float(rng.exponential(20.0)),
                    "log1p_hours": float(np.log1p(rng.exponential(20.0))),
                    "project_primary_genre": f"G{appid % 3}",
                    "is_free": int(appid == 1),
                    "is_text_canonical": True,
                    "not_short_or_low_information": True,
                    "first_author_review": g == 0,
                    "complete_game": True,
                })
                weights_rows.append({
                    "review_id": rid,
                    "T01": t01, "T02": 0.0, "T03": t03, "T04": 0.0,
                    "T05": t05, "T06": 0.0, "T07": 0.0, "T08": 0.0,
                })
        master = pd.DataFrame(rows).set_index("review_id")
        weights = pd.DataFrame(weights_rows).set_index("review_id")
        return master, weights

    def test_feasibility_counts_and_decision(self):
        master, weights = self._synthetic_data()
        summary, dist, temporal, decision = rq5.compute_feasibility(master, weights)
        elig = summary.set_index("cohort").loc["association_eligible"]
        self.assertEqual(int(elig["reviews"]), 70)
        self.assertEqual(int(elig["unique_authors"]), 60)
        self.assertEqual(int(elig["single_game_authors"]), 50)
        self.assertEqual(int(elig["authors_ge2_games"]), 10)
        self.assertEqual(decision["selected_task"], "recommendation_label_prediction")
        self.assertEqual(int(dist.query("cohort == 'association_eligible'")["author_count"].sum()), 60)
        self.assertEqual(len(temporal.query("cohort == 'association_eligible'")), 3)

    def test_cv_no_author_leakage_and_majority_baseline(self):
        master, weights = self._synthetic_data()
        df = rq5.prepare_cohort_frame(master, weights, only_covered=True)
        y = df["recommend"].to_numpy(dtype=int)
        groups = df["steamid"].to_numpy()

        pred_maj, prob_maj, _ = rq5.fit_predict_cv(df, y, groups, ["appid_str"], [], model_kind="majority", n_splits=5)
        pts_maj = rq5.point_metrics(y, pred_maj, prob_maj)
        self.assertAlmostEqual(pts_maj["balanced_accuracy"], 0.5, places=6)

        pred_base, prob_base, _ = rq5.fit_predict_cv(
            df, y, groups, ["appid_str", "language"], ["log1p_hours"], model_kind="logreg", n_splits=5,
        )
        pred_top, prob_top, _ = rq5.fit_predict_cv(
            df, y, groups, ["appid_str", "language"], ["log1p_hours"] + list(rq5.TOPICS), model_kind="logreg", n_splits=5,
        )
        stats = rq5.bootstrap_metrics_and_gain(y, pred_top, prob_top, groups, pred_base, prob_base, n_boot=200)
        self.assertGreater(stats["balanced_accuracy"], stats["balanced_accuracy"] - stats["delta_bal_acc"])
        self.assertGreater(stats["delta_bal_acc"], 0.05)


if __name__ == "__main__":
    unittest.main()
