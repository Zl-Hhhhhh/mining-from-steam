import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import adjudicate_review_round1 as adjudicate
import review_topics_json as review


class AdjudicationTests(unittest.TestCase):
    def test_drafts_stay_unsigned_and_ai_completion_is_not_human_agreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "final"
            result, audit = adjudicate.run(output)
            self.assertTrue(result["complete"])
            self.assertFalse(result["human_adjudication_complete"])
            self.assertFalse(result["contributors_complete"])
            self.assertTrue(all(not x["submitted_acknowledged_review"] for x in result["contributors"]))
            self.assertEqual((len(result["cases"]), len(result["groups"]), len(audit)), (180, 30, 210))
            resolved = review.read_json(output / "resolved_sample.json")
            self.assertEqual(len(resolved["cases"]), 300)
            self.assertEqual(sum(x["origin"] == "original_label_agreement" for x in resolved["cases"].values()), 120)
            self.assertEqual(result["groups"]["G10"]["related_groups"], ["G23"])
            self.assertEqual(result["groups"]["G23"]["related_groups"], ["G10"])
            self.assertTrue(all(x["final_answer"]["coherence_rating"] == 0 for x in result["groups"].values()
                                if x["final_answer"]["group_kind"] == "generic_evaluation"))
            original = review.read_json(adjudicate.BASE / "review_results.json")
            source = review.ROOT / "docs/review"
            imported = review.run(source, source / "review_sample.json", source / "group_review_guidance.json",
                                  Path(temporary) / "imported", output / "adjudication.json")
            stored = review.read_json(Path(temporary) / "imported/review_results.json")
            self.assertEqual(stored["metrics"], original["metrics"])
            self.assertFalse(imported["human_adjudication_complete"])
            self.assertFalse(imported["task12_topics_frozen"])
            self.assertEqual(imported["review_session_id"], review.sha(output / "adjudication.json"))
            with self.assertRaises(FileExistsError):
                adjudicate.run(output)

    def test_pending_draft_answers_and_assignment_identity_are_validated(self):
        read = review.read_json
        for invalid in ("assignment", "answer", "coverage"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                def altered(path):
                    data = read(path)
                    if Path(path).name.endswith("B_Lin_draft.json"):
                        data = copy.deepcopy(data)
                        if invalid == "assignment":
                            data["assignment_sha256"] = "wrong"
                        elif invalid == "answer":
                            next(iter(data["cases"].values()))["final_answer"] = None
                        else:
                            data["cases"].pop(next(iter(data["cases"])))
                    return data
                with patch.object(review, "read_json", side_effect=altered):
                    with self.assertRaises(ValueError):
                        adjudicate.run(Path(temporary) / "final")


if __name__ == "__main__":
    unittest.main()
