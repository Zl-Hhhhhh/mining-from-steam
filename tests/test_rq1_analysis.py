import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_rq1


class RQ1AggregationTests(unittest.TestCase):
    def setUp(self):
        self.names = {
            topic_id: {"topic_id": topic_id, "name_zh": topic_id, "name_en": topic_id}
            for topic_id in analyze_rq1.TOPIC_IDS
        }

    def test_group_share_uses_topic_evidence_denominator(self):
        rows = []
        for review_id, topic_id in [("1", "T01"), ("2", "T02"), ("3", None)]:
            row = {"group": "g", "fit_T01": 0.0, "fit_T02": 0.0}
            for topic in analyze_rq1.TOPIC_IDS[2:]:
                row[f"fit_{topic}"] = 0.0
            if topic_id:
                row[f"fit_{topic_id}"] = 1.0
            row["fit_topic_total"] = sum(row[f"fit_{topic}"] for topic in analyze_rq1.TOPIC_IDS)
            row["fit_has_topic"] = row["fit_topic_total"] > 0
            rows.append(row)
        frame = pd.DataFrame(rows)
        result = analyze_rq1.aggregate_topics(frame, ["group"], "fit", self.names)
        self.assertEqual(result.topic_evidence_reviews.iloc[0], 2)
        self.assertEqual(result.no_topic_evidence_reviews.iloc[0], 1)
        self.assertAlmostEqual(result.loc[result.topic_id == "T01", "topic_share"].iloc[0], 0.5)
        self.assertAlmostEqual(result.loc[result.topic_id == "T02", "topic_share"].iloc[0], 0.5)

    def test_empty_group_has_zero_share(self):
        frame = pd.DataFrame([{
            "fit_topic_total": 0.0,
            "fit_has_topic": False,
            **{f"fit_{topic}": 0.0 for topic in analyze_rq1.TOPIC_IDS},
        }])
        result = analyze_rq1.aggregate_topics(frame, [], "fit", self.names)
        self.assertTrue((result.topic_share == 0).all())

    def test_representatives_are_unique_per_review(self):
        frame = pd.DataFrame([{
            "recommendationid": "1", "appid": 7, "game": "Game", "language": "english",
            "language_label": "English", "fit_T01": 1.0,
        }])
        assignments = [{
            "assignment_scope": "fit_candidate", "topic_ids": ["T01"],
            "topic_scores": {"T01": 0.9}, "review_ids": ["1"],
            "confidence": "high", "text": "mechanics",
        }, {
            "assignment_scope": "fit_candidate", "topic_ids": ["T01"],
            "topic_scores": {"T01": 0.8}, "review_ids": ["1"],
            "confidence": "high", "text": "mechanics again",
        }]
        result = analyze_rq1.representative_comments(frame, assignments, self.names)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.review_id.iloc[0], "1")


if __name__ == "__main__":
    unittest.main()
