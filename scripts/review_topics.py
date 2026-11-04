"""Compare two completed independent review sheets; never invent human ratings."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from analysis_common import save_json, sha256, write_csv


def compare_reviews(first_path, second_path, destination):
    frames = []
    for path in (first_path, second_path):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if frame.review_id.duplicated().any():
            raise ValueError(f"Duplicate review IDs: {path}")
        if not frame.theme_label.str.strip().ne("").all():
            raise ValueError(f"Incomplete independent labels: {path}. Blank sheets are not completed human review.")
        frames.append(frame)
    first, second = frames
    if set(first.review_id) != set(second.review_id):
        raise ValueError("Both raters must label the same reviews")
    merged = first.merge(second, on="review_id", suffixes=("_A", "_B"), validate="one_to_one")
    if not (merged.review_text_A == merged.review_text_B).all():
        raise ValueError("Review texts differ between rater sheets")
    if not (merged.language_A == merged.language_B).all():
        raise ValueError("Languages differ between rater sheets")
    # A/B must agree on the label vocabulary before independent annotation.
    # Do not map independently invented topic IDs to one another post hoc.
    result = {"status": "two_completed_sheets_compared_not_team_topic_freeze",
              "sheets": [{"path": str(p), "sha256": sha256(p)} for p in (first_path, second_path)],
              "rows": len(merged), "by_language": {}}
    for language, group in merged.groupby("language_A"):
        left, right = group.theme_label_A.str.strip(), group.theme_label_B.str.strip()
        kappa = cohen_kappa_score(left, right) if len(set(left) | set(right)) > 1 else None
        result["by_language"][language] = {"rows": len(group), "agreement": float((left == right).mean()),
                                             "cohen_kappa": None if kappa is None else float(kappa)}
    disagreements = merged[merged.theme_label_A.str.strip() != merged.theme_label_B.str.strip()]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "disagreements.csv", disagreements)
    save_json(destination / "agreement.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_reviews(args.reviewer_a, args.reviewer_b, args.output)
    print(f"Compared {result['rows']} independently annotated reviews")
