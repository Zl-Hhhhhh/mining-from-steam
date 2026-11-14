"""Scientific and provenance safeguards for the Task11 JSON comparison."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_topics_json as review


class JsonReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = review.read_json(review.ROOT / "docs/review/review_sample.json")

    def fixture(self):
        sample = copy.deepcopy(self.sample)
        sample["cases"] = [{"id": "R001", "text": "lag"}, {"id": "R002", "text": "story"}]
        sample["groups"] = [{"id": "G01", "examples": [{"id": "G01-E1", "kind": "center", "text": "lag"}]}]
        answer = {"observed_language": "english", "aspect_presence": "yes", "aspect_labels": ["performance_stability"],
                  "aspect_evidence": {"performance_stability": "lag"}, "reason": "", "expression_status": "ordinary",
                  "expression_flags": [], "style_evidence": "", "notes": ""}
        group = {"group_kind": "single_aspect", "coherence_rating": 2, "proposed_name": "Runtime", "notes": "E1 lag"}
        submissions = {}
        for rater in ("A", "B"):
            for stage in ("cases", "groups"):
                submissions[(rater, stage)] = {"schema_version": "1.0", "package_id": sample["package_id"],
                    "sample_sha256": sample["sample_sha256"], "rater_id": rater, "stage": stage, "rater_name": rater,
                    "complete": True, "acknowledged_codebook": True, "stage1_import_confirmed": stage == "groups",
                    "answers": {i["id"]: copy.deepcopy(answer if stage == "cases" else group) for i in sample[stage]}}
        return sample, submissions

    def test_duplicate_json_keys_are_rejected_instead_of_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"answers":{"R001":{},"R001":{}}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                review.read_json(path)

    def test_metadata_ids_false_completion_and_invalid_answers_are_rejected(self):
        for change in ("package", "ids", "complete", "gate", "answer", "rater"):
            with self.subTest(change=change):
                sample, data = self.fixture()
                d = data[("A", "groups" if change == "gate" else "cases")]
                if change == "package": d["sample_sha256"] = "changed"
                if change == "ids": d["answers"].pop("R001")
                if change == "complete": d["complete"] = False
                if change == "gate": d["stage1_import_confirmed"] = False
                if change == "answer": d["answers"]["R001"]["aspect_evidence"] = {}
                if change == "rater": d["rater_id"] = "B"
                with self.assertRaises(ValueError):
                    review.validate_submissions(sample, data)

    def test_kappa_matches_independent_library_and_degenerate_values_are_null(self):
        pairs = [(0, 0), (0, 1), (1, 1), (1, 1), (2, 0), (2, 2)]
        result = review.agreement(pairs, weighted=True)
        self.assertAlmostEqual(result["cohen_kappa"], cohen_kappa_score(*zip(*pairs)))
        self.assertAlmostEqual(result["quadratic_weighted_kappa"], cohen_kappa_score(*zip(*pairs), weights="quadratic"))
        self.assertIsNone(review.agreement([(1, 1)] * 4)["cohen_kappa"])
        self.assertIsNone(review.agreement([])["agreement"])
        json.dumps(review.agreement([]), allow_nan=False)

    def test_uncertainty_is_not_encoded_as_negative_and_denominators_are_explicit(self):
        sample, data = self.fixture()
        a = data[("A", "cases")]["answers"]["R002"]
        a.update(aspect_presence="uncertain", aspect_labels=[], reason="missing_context", notes="Context absent")
        rows = review.make_case_records(sample, data)
        result = review.case_metrics(rows, sample)
        self.assertEqual(result["aspect_labels"]["performance_stability"]["n"], 1)
        self.assertEqual(result["aspect_labels"]["performance_stability"]["excluded_uncertain"], 1)
        self.assertTrue(rows[1]["requires_review"])
        self.assertEqual(result["aspect_presence"]["n"], 2)

    def test_array_order_and_evidence_wording_are_not_label_disagreements(self):
        sample, data = self.fixture()
        a = data[("A", "cases")]["answers"]["R001"]
        a["aspect_labels"].append("gameplay")
        a["aspect_evidence"]["gameplay"] = "lag"
        b = copy.deepcopy(a)
        b["aspect_labels"].reverse()
        b["notes"] = "Different note"
        b["reason"] = "generic_evaluation"  # Inactive stale reason from a prior edit.
        self.assertEqual(review.differences(a, b), [])

    def test_shared_group_error_and_nonexistent_reference_still_require_review(self):
        sample, data = self.fixture()
        for rater in ("A", "B"):
            data[(rater, "groups")]["answers"]["G01"].update(group_kind="generic_evaluation", notes="E16 fun")
        row = review.make_group_records(sample, data, {})[0]
        self.assertEqual(row["differences"], [])
        self.assertEqual(row["priority"], "high")
        self.assertTrue(row["requires_review"])
        self.assertEqual(len(row["checks"]), 4)

    def test_draft_is_not_a_completed_adjudication_and_changed_sources_are_rejected(self):
        sample, data = self.fixture()
        result = review.analyse(sample, data, {})
        result["source_fingerprint"] = "test-source"
        draft = review.make_template(result)
        self.assertEqual(review.decision_errors(draft, result, sample), [])
        draft["complete"] = True
        self.assertTrue(review.decision_errors(draft, result, sample))
        draft["complete"] = False
        draft["source_fingerprint"] = "different"
        self.assertIn("source_fingerprint 不匹配", review.decision_errors(draft, result, sample))

    def test_modified_adoption_must_be_marked_as_custom(self):
        sample, data = self.fixture()
        data[("B", "cases")]["answers"]["R001"]["observed_language"] = "other"
        result = review.analyse(sample, data, {})
        result["source_fingerprint"] = "test-source"
        draft = review.make_template(result)
        final = copy.deepcopy(data[("A", "cases")]["answers"]["R001"])
        final["observed_language"] = "schinese"
        draft["cases"]["R001"].update(status="reviewed", basis="A", final_answer=final, rationale="Modified language")
        self.assertTrue(review.decision_errors(draft, result, sample))
        draft["cases"]["R001"]["basis"] = "custom"
        self.assertEqual(review.decision_errors(draft, result, sample), [])

    def test_real_comparison_preserves_inputs_and_rejects_output_overwrite(self):
        input_dir = review.ROOT / "docs/review"
        paths = sorted(input_dir.glob("*_complete.json"))
        before = {p.name: review.sha(p) for p in paths}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison"
            result = review.run(input_dir, input_dir / "review_sample.json", input_dir / "group_review_guidance.json", output)
            self.assertEqual(result["counts"]["cases_total"], 300)
            self.assertEqual(result["counts"]["groups_total"], 30)
            self.assertTrue(all(r["status"] == "pending" for r in result["initial_decisions"]["groups"].values()))
            self.assertFalse(result["human_adjudication_complete"])
            self.assertFalse(result["task12_topics_frozen"])
            self.assertEqual({p.name: review.sha(p) for p in paths}, before)
            with self.assertRaises(FileExistsError):
                review.run(input_dir, input_dir / "review_sample.json", input_dir / "group_review_guidance.json", output)


if __name__ == "__main__":
    unittest.main()
