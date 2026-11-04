"""Meaningful safeguards for unsupervised fitting and independent topic review."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analysis_common import ROOT, hash_status, normalize_weights, read_json
from nmf_baseline import ReviewTokenizer, get_assignments, sample_for_review
from review_topics import compare_reviews


class TopicAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(ROOT / "config/nmf_baseline_aspect.json")

    def test_configuration_has_valid_chinese_not_encoding_placeholders(self):
        self.assertIn("好玩", self.config["chinese_stopwords"])
        self.assertFalse(any("?" in x for x in self.config["chinese_stopwords"]))

    def test_negation_survives_english_and_chinese_tokenization(self):
        english = ReviewTokenizer("english", self.config, ["Counter-Strike 2"])
        chinese = ReviewTokenizer("schinese", self.config, ["Counter-Strike 2"])
        self.assertIn("not", english("Can't recommend combat; don't buy Counter Strike"))
        self.assertIn("not", english("The combat is not smooth"))
        self.assertNotIn("counter", english("Counter Strike combat"))
        self.assertIn("剧情", chinese("剧情没有优化，也不好玩"))
        self.assertTrue({"没", "没有"} & set(chinese("剧情没有优化，也不好玩")))

    def test_component_rescaling_does_not_change_topic_proportions(self):
        weights = np.array([[1., 2.], [0., 0.]])
        components = np.array([[1., 2., 0.], [0., 2., 1.]])
        scale = np.array([7., 0.1])
        np.testing.assert_allclose(normalize_weights(weights, components),
                                   normalize_weights(weights / scale, components * scale[:, None]))
        self.assertTrue(np.isfinite(normalize_weights(weights, components)).all())

    def test_zero_vectors_and_low_information_are_not_forced_into_a_theme(self):
        frame = pd.DataFrame({"is_low_information": [False, True, False], "voted_up": [True, False, True]})
        matrix = csr_matrix([[1., 1.], [1., 1.], [0., 0.]])
        weights = np.array([[1., 0.], [1., 0.], [0., 0.]])
        components = np.eye(2)
        result = get_assignments(weights, matrix, components, frame, self.config)
        self.assertEqual(result[1][-1], -1)
        self.assertEqual(result[-1].tolist(), [True, False, False])
        frame.voted_up = ~frame.voted_up
        changed = get_assignments(weights, matrix, components, frame, self.config)
        for left, right in zip(result, changed):
            np.testing.assert_array_equal(left, right)

    def test_hash_accepts_line_endings_but_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            expected = hashlib.sha256(b'{"id":1}\n').hexdigest()
            path.write_bytes(b'{"id":1}\r\n')
            self.assertEqual(hash_status(path, expected), "LF-equivalent")
            path.write_bytes(b'{"id":2}\n')
            with self.assertRaises(ValueError):
                hash_status(path, expected)

    def test_review_sample_is_blind_unique_and_contains_no_fabricated_labels(self):
        frame = pd.DataFrame([{"recommendationid": str(i), "appid": i % 4, "language": lang,
                               "review_clean": f"review text {i}", "voted_up": i % 2 == 0,
                               "is_short_text": False}
                              for i, lang in enumerate(["english"] * 12 + ["schinese"] * 12)])
        assignment = frame[["recommendationid", "appid", "language"]].copy()
        assignment["clear_theme"] = True
        assignment["topic_index"] = np.arange(len(frame)) % 2
        assignment["cosine"] = np.linspace(0.1, 0.9, len(frame))
        assignment["topic_id"] = assignment.language + ":T" + assignment.topic_index.astype(str)
        config = {**self.config, "review_sample_size": 20}
        blind, key = sample_for_review(frame, assignment, config)
        self.assertEqual(len(blind), 20)
        self.assertTrue(key.recommendationid.is_unique)
        self.assertTrue(blind.theme_label.eq("").all())
        self.assertTrue(blind.no_clear_theme.eq("").all())
        self.assertFalse({"voted_up", "steamid", "candidate_theme"} & set(blind.columns))

    def test_empty_review_sheets_cannot_be_reported_as_completed_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rater.csv"
            pd.DataFrame({"review_id": ["R001"], "review_text": ["story"],
                          "language": ["english"], "theme_label": [""]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                compare_reviews(path, path, Path(directory) / "result")

    def test_agreement_reports_actual_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.csv", Path(directory) / "b.csv"
            frame = pd.DataFrame({"review_id": ["1", "2", "3", "4"], "review_text": ["a", "b", "c", "d"],
                                  "language": ["english"] * 4, "theme_label": ["story", "story", "combat", "combat"]})
            frame.to_csv(first, index=False)
            frame.loc[3, "theme_label"] = "story"
            frame.to_csv(second, index=False)
            result = compare_reviews(first, second, Path(directory) / "result")
            self.assertEqual(result["by_language"]["english"]["agreement"], 0.75)
            self.assertAlmostEqual(result["by_language"]["english"]["cohen_kappa"], 0.5)


if __name__ == "__main__":
    unittest.main()
