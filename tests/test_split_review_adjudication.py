"""Scope, coverage and handoff safeguards for parallel human adjudication."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_topics_json as review
import split_review_adjudication as split


class SplitAdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = split.load_parent(split.DEFAULT_REVIEW)
        cls.sample = review.read_json(review.ROOT / "docs/review/review_sample.json")

    def completed_part(self, part):
        data = split.scoped_payload(self.result, part)["initial_decisions"]
        data.update(complete=True, reviewer="TEST ONLY", acknowledged_review=True)
        cases = {r["id"]: r for r in self.result["cases"]}
        groups = {r["id"]: r for r in self.result["groups"]}
        for i, row in data["cases"].items():
            row.update(status="reviewed", basis="A", final_answer=copy.deepcopy(cases[i]["A"]), rationale="Synthetic test")
        for i, row in data["groups"].items():
            row.update(status="reviewed", action="uncertain", related_groups=[], final_answer=copy.deepcopy(groups[i]["A"]), rationale="Synthetic test")
        return data

    def test_exact_disjoint_coverage_and_balanced_priority_quotas(self):
        parts = split.partition(self.result)
        split.verify_partition(self.result, parts)
        self.assertEqual([(len(p["case_ids"]), len(p["group_ids"])) for p in parts], [(90, 15), (90, 15)])
        self.assertEqual([p["high_priority_cases"] for p in parts], [64, 64])
        self.assertEqual(sorted(p["high_priority_groups"] for p in parts), [8, 9])
        self.assertLess(max(p["estimated_effort"] for p in parts) / min(p["estimated_effort"] for p in parts), 1.1)
        self.assertEqual(parts, split.partition(self.result))

    def test_payload_keeps_full_readonly_group_context_and_empty_assigned_decisions(self):
        part = split.partition(self.result)[0]
        data = split.scoped_payload(self.result, part)
        self.assertEqual(len(data["group_reference"]), 30)
        self.assertEqual(data["group_reference"], self.result["groups"])
        self.assertEqual(set(data["initial_decisions"]["cases"]), set(part["case_ids"]))
        self.assertEqual(len(data["cases"]), 90)
        self.assertEqual(len(data["groups"]), 15)
        self.assertEqual(data["metrics"], self.result["metrics"])
        self.assertTrue(all(r["status"] == "pending" for r in data["initial_decisions"]["cases"].values()))

    def test_wrong_scope_assignment_source_missing_or_extra_items_are_rejected(self):
        parts = split.partition(self.result)
        for change in ("scope", "assignment", "checksum", "source", "missing", "extra", "incomplete"):
            with self.subTest(change=change):
                data = self.completed_part(parts[0])
                if change == "scope": data["scope"] = "full"
                if change == "assignment": data["assignment_id"] = parts[1]["id"]
                if change == "checksum": data["assignment_sha256"] = "wrong"
                if change == "source": data["source_fingerprint"] = "wrong"
                if change == "missing": data["cases"].pop(parts[0]["case_ids"][0])
                if change == "extra": data["cases"][parts[1]["case_ids"][0]] = next(iter(data["cases"].values()))
                if change == "incomplete": data["complete"] = False
                self.assertTrue(split.assignment_errors(data, parts[0], self.result, self.sample))

    def test_completed_part_cannot_be_imported_as_full_round(self):
        part = split.partition(self.result)[0]
        data = self.completed_part(part)
        self.assertEqual(split.assignment_errors(data, part, self.result, self.sample), [])
        self.assertTrue(review.decision_errors(data, self.result, self.sample))

    def test_manifest_overlap_and_missing_ids_are_rejected(self):
        parts = split.partition(self.result)
        parts[1]["case_ids"][0] = parts[0]["case_ids"][0]
        with self.assertRaisesRegex(ValueError, "overlap or omit"):
            split.verify_partition(self.result, parts)

    def test_standalone_pages_and_merge_preserve_inputs_and_require_coordinator_confirmation(self):
        sources = {r["filename"]: r["sha256"] for r in self.result["sources"]}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = split.split(split.DEFAULT_REVIEW, directory / "assignments")
            first, second = manifest["assignments"]
            for part in (first, second):
                page = directory / "assignments" / part["review_file"]
                self.assertTrue(page.exists())
                self.assertEqual(review.sha(page), part["review_file_sha256"])
            self.assertEqual(len(list((directory / "assignments").iterdir())), 3)
            a, b = self.completed_part(first), self.completed_part(second)
            group_id, related_id = first["group_ids"][0], second["group_ids"][0]
            a["groups"][group_id].update(action="merge", related_groups=[related_id])
            self.assertEqual(split.assignment_errors(a, first, self.result, self.sample), [])
            review.write_json(directory / "a.json", a)
            review.write_json(directory / "b.json", b)
            merged = split.merge(directory / "assignments", directory / "a.json", directory / "b.json", directory / "merged")
            self.assertEqual(len(merged["cases"]), 180)
            self.assertEqual(len(merged["groups"]), 30)
            self.assertFalse(merged["complete"])
            self.assertFalse(merged["acknowledged_review"])
            self.assertEqual(merged["reviewer"], "")
            self.assertTrue(merged["coordinator_confirmation_required"])
            self.assertTrue(merged["contributors_complete"])
            self.assertEqual(len(merged["cross_part_links_to_review"]), 1)
            self.assertEqual(review.decision_errors(merged, self.result, self.sample), [])
            self.assertTrue((directory / "merged/coordinator_review/人工复核.html").exists())
            with self.assertRaises(FileExistsError):
                split.split(split.DEFAULT_REVIEW, directory / "assignments")
        for name, expected in sources.items():
            self.assertEqual(review.sha(review.ROOT / "docs/review" / name), expected)


if __name__ == "__main__":
    unittest.main()
