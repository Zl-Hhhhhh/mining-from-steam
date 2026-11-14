import copy
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recover_review_group_sources as recovery
import review_topics_json as review


class SourceRecoveryTests(unittest.TestCase):
    def test_out_of_scope_placeholder_is_not_a_cluster(self):
        group = {"examples": [{"text": "one"}, {"text": "two"}]}
        ids = {"one": {"a"}, "two": {"b"}}
        self.assertEqual(recovery.compatible_topics(group, ids, {"a": -1, "b": -1}), set())
        self.assertEqual(recovery.compatible_topics(group, ids, {"a": 0, "b": 0}), {0})

    def test_joint_anchor_must_be_unique_and_keep_groups_distinct(self):
        groups = [{"id": "G01"}, {"id": "G02"}]
        candidate = {"family": "nmf_english", "run_id": "run1"}
        matches = {"G01": {"run1": {0}}, "G02": {"run1": {1}}}
        chosen, topics = recovery.choose_anchor(groups, [candidate], matches, {}, "nmf_english")
        self.assertEqual(chosen, candidate)
        self.assertEqual(set(topics.values()), {0, 1})
        ambiguous = copy.deepcopy(matches)
        for options in ambiguous.values():
            options["run2"] = options["run1"]
        with self.assertRaisesRegex(ValueError, "found 2"):
            recovery.choose_anchor(groups, [candidate, {**candidate, "run_id": "run2"}], ambiguous, {}, "nmf_english")
        with self.assertRaisesRegex(ValueError, "found 0"):
            recovery.choose_anchor(groups, [candidate], {"G01": {"run1": {0}}, "G02": {"run1": {0}}}, {}, "nmf_english")

    def test_bge_center_evidence_disambiguates_membership_matches(self):
        groups = [{"id": "G01", "examples": [{"kind": "center", "text": "center text"}]}]
        candidates = [{"family": "bge_shared", "run_id": "run1", "candidate": "first"},
                      {"family": "bge_shared", "run_id": "run2", "candidate": "second"}]
        matches = {"G01": {"run1": {0}, "run2": {2}}}
        centers = defaultdict(set, {("first", 0): {"center text"}})
        chosen, _ = recovery.choose_anchor(groups, candidates, matches, centers, "bge_shared")
        self.assertEqual(chosen["run_id"], "run1")

    def test_real_recovery_has_full_evidence_and_retains_duplicate_id_ambiguity(self):
        before = review.sha(recovery.SAMPLE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "recovered"
            result = recovery.recover(output)
            self.assertEqual(len(result["groups"]), 30)
            self.assertEqual(sum(len(g["examples"]) for g in result["groups"]), 180)
            self.assertEqual(len(result["anchors"]), 3)
            self.assertTrue(all(a["k"] == 10 and a["seed"] == 17 for a in result["anchors"]))
            self.assertFalse(result["original_coordinator_key_found"])
            self.assertFalse(result["task12_topics_frozen"])
            self.assertTrue(all(e["matching_ids_in_anchor_topic"] for g in result["groups"] for e in g["examples"]))
            self.assertTrue(any(not e["source_id_unique"] for g in result["groups"] for e in g["examples"]))
            self.assertTrue(all(e["published_bge_center_match"] for g in result["groups"] if g["family"] == "bge_shared"
                                for e in g["examples"] if e["kind"] == "center"))
            self.assertTrue(all(review.sha(recovery.ROOT / p["path"]) == p["sha256"] for p in result["sources"]))
            with self.assertRaises(FileExistsError):
                recovery.recover(output)
        self.assertEqual(review.sha(recovery.SAMPLE), before)


if __name__ == "__main__":
    unittest.main()
