"""Distribute disjoint Task11 decisions, then collect them for coordinator review."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import review_topics_json as review

DEFAULT_REVIEW = review.ROOT / "output/topic_review/steam-review-20261107-v1"
OWNERS = (("A_Cai", "A-Cai"), ("B_Lin", "B-Lin"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def effort(row, stage):
    text = row["text"] if stage == "cases" else "".join(e["text"] for e in row["examples"])
    evidence = sum(len(json.dumps(row[r], ensure_ascii=False)) for r in ("A", "B"))
    return len(text) + evidence + 100 * len(row["differences"]) + (300 if stage == "groups" else 0)


def partition(result):
    """Largest-work-first within priority strata, with exact item-count quotas."""
    parts = [{"id": i, "label": label, "case_ids": [], "group_ids": [], "estimated_effort": 0}
             for i, label in OWNERS]
    for stage, key in (("cases", "case_ids"), ("groups", "group_ids")):
        rows = [r for r in result[stage] if r["requires_review"]]
        quotas = [len(rows) // 2, len(rows) - len(rows) // 2]
        for priority in ("high", "normal"):
            selected = sorted((r for r in rows if r["priority"] == priority),
                              key=lambda r: (-effort(r, stage), r["id"]))
            low = min(range(2), key=lambda p: (parts[p]["estimated_effort"], p))
            cap = [len(selected) // 2] * 2
            cap[low] += len(selected) % 2
            used = [0, 0]
            if priority == "normal":
                cap = [quotas[p] - len(parts[p][key]) for p in range(2)]
            for row in selected:
                candidates = [p for p in range(2) if used[p] < cap[p]]
                p = min(candidates, key=lambda x: (parts[x]["estimated_effort"], len(parts[x][key]), x))
                parts[p][key].append(row["id"])
                parts[p]["estimated_effort"] += effort(row, stage)
                used[p] += 1
        assert [len(p[key]) for p in parts] == quotas
    for part in parts:
        part["case_ids"].sort()
        part["group_ids"].sort()
        part["high_priority_cases"] = sum(r["id"] in part["case_ids"] and r["priority"] == "high" for r in result["cases"])
        part["high_priority_groups"] = sum(r["id"] in part["group_ids"] and r["priority"] == "high" for r in result["groups"])
        part["assignment_sha256"] = digest({"source_fingerprint": result["source_fingerprint"],
                                             "id": part["id"], "case_ids": part["case_ids"], "group_ids": part["group_ids"]})
    return parts


def verify_partition(result, parts):
    if [p["id"] for p in parts] != [x[0] for x in OWNERS]:
        raise ValueError("Assignment owners differ from the expected two parts")
    for stage, key in (("cases", "case_ids"), ("groups", "group_ids")):
        expected = {r["id"] for r in result[stage] if r["requires_review"]}
        first, second = (set(p[key]) for p in parts)
        if first & second or first | second != expected:
            raise ValueError(f"{stage}: assignments overlap or omit required items")
        if any(len(p[key]) != len(set(p[key])) for p in parts):
            raise ValueError(f"{stage}: duplicate assignment IDs")
    for part in parts:
        expected = digest({"source_fingerprint": result["source_fingerprint"], "id": part["id"],
                           "case_ids": part["case_ids"], "group_ids": part["group_ids"]})
        if part["assignment_sha256"] != expected:
            raise ValueError("Assignment checksum mismatch")


def load_parent(directory):
    directory = Path(directory)
    result = review.read_json(directory / "review_results.json")
    for source in result["sources"]:
        path = review.ROOT / "docs/review" / source["filename"]
        if review.sha(path) != source["sha256"]:
            raise ValueError(f"Source changed since comparison: {source['filename']}")
    return result


def scoped_payload(result, part):
    payload = copy.deepcopy(result)
    payload["assignment"] = copy.deepcopy(part)
    payload["parent_report_template_sha256"] = result["report_template_sha256"]
    payload["report_template_sha256"] = review.sha(Path(review.__file__).with_suffix(".html"))
    payload["splitter_sha256"] = review.sha(Path(__file__))
    payload["population_counts"] = {"cases": result["counts"]["cases_total"], "groups": len(result["groups"])}
    payload["group_reference"] = copy.deepcopy(result["groups"])
    payload["cases"] = [r for r in payload["cases"] if r["id"] in part["case_ids"]]
    payload["groups"] = [r for r in payload["groups"] if r["id"] in part["group_ids"]]
    payload["counts"].update(cases_total=len(payload["cases"]), cases_requiring_review=len(payload["cases"]),
                             cases_high_priority=part["high_priority_cases"], groups_total=len(payload["groups"]),
                             groups_high_priority=part["high_priority_groups"])
    seed = copy.deepcopy(result["initial_decisions"])
    seed.update(scope="assignment", assignment_id=part["id"], assignment_sha256=part["assignment_sha256"],
                complete=False, reviewer="", acknowledged_review=False)
    seed["cases"] = {i: copy.deepcopy(seed["cases"][i]) for i in part["case_ids"]}
    seed["groups"] = {i: copy.deepcopy(seed["groups"][i]) for i in part["group_ids"]}
    payload["initial_decisions"] = seed
    return payload


def render(payload, destination):
    template = Path(review.__file__).with_suffix(".html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    destination.write_text(template.replace("__REVIEW_PAYLOAD__", data), encoding="utf-8", newline="\n")


def split(review_dir, destination):
    review_dir, destination = Path(review_dir), Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new assignment output directory")
    result = load_parent(review_dir)
    parts = partition(result)
    verify_partition(result, parts)
    manifest = {"schema_version": "1.0", "package_id": result["package_id"],
                "sample_sha256": result["sample_sha256"], "source_fingerprint": result["source_fingerprint"],
                "parent_review_results_sha256": review.sha(review_dir / "review_results.json"),
                "generated_at": datetime.now(timezone.utc).isoformat(), "assignments": parts,
                "splitter_sha256": review.sha(Path(__file__)),
                "html_template_sha256": review.sha(Path(review.__file__).with_suffix(".html")),
                "allocation_method": "Priority-stratified longest-work-first, exact 90/15 quotas per person; effort = text and answer characters + 100 per differing field + 300 per group",
                "coordinator_confirmation_required_after_merge": True}
    destination.mkdir(parents=True)
    for part in parts:
        payload = scoped_payload(result, part)
        page = destination / f"{part['label']}_人工复核.html"
        render(payload, page)
        part["review_file"] = page.name
        part["review_file_sha256"] = review.sha(page)
    review.write_json(destination / "assignment_manifest.json", manifest)
    return manifest


def assignment_errors(data, part, result, sample):
    errors = []
    if data.get("scope") != "assignment" or data.get("assignment_id") != part["id"]:
        errors.append("分任务编号不匹配")
    if data.get("assignment_sha256") != part["assignment_sha256"]:
        errors.append("分任务来源校验不匹配")
    for stage, key in (("cases", "case_ids"), ("groups", "group_ids")):
        if not isinstance(data.get(stage), dict) or set(data[stage]) != set(part[key]):
            errors.append(f"{stage} 必须恰好包含本份分配的编号")
    if errors:
        return errors
    # Apply the full answer/schema validation while keeping completion local to this part.
    check = {**data, "scope": "full", "complete": False}
    errors.extend(review.decision_errors(check, result, sample))
    if data.get("complete") is not True or data.get("acknowledged_review") is not True or not str(data.get("reviewer", "")).strip():
        errors.append("本份尚未确认完成或缺少处理人")
    for stage in ("cases", "groups"):
        if any(row.get("status") != "reviewed" for row in data[stage].values()):
            errors.append(f"{stage} 仍有未确认项")
    return errors


def merge(assignments_dir, first, second, destination, review_dir=DEFAULT_REVIEW):
    assignments_dir, destination, review_dir = map(Path, (assignments_dir, destination, review_dir))
    if destination.exists():
        raise FileExistsError("Use a new merged-review directory")
    manifest = review.read_json(assignments_dir / "assignment_manifest.json")
    result = load_parent(review_dir)
    if manifest["parent_review_results_sha256"] != review.sha(review_dir / "review_results.json") or manifest["source_fingerprint"] != result["source_fingerprint"]:
        raise ValueError("Assignment manifest belongs to another comparison")
    verify_partition(result, manifest["assignments"])
    sample = review.read_json(review.ROOT / "docs/review/review_sample.json")
    combined = review.make_template(result)
    contributors, owner_by_group = [], {}
    for path, part in zip((Path(first), Path(second)), manifest["assignments"]):
        data = review.read_json(path)
        errors = assignment_errors(data, part, result, sample)
        if errors:
            raise ValueError(f"{part['label']}: " + "; ".join(errors))
        for stage in ("cases", "groups"):
            combined[stage].update(copy.deepcopy(data[stage]))
        contributors.append({"assignment_id": part["id"], "reviewer": data["reviewer"],
                             "file_sha256": review.sha(path), "acknowledged_review": True})
        owner_by_group.update({g: part["id"] for g in part["group_ids"]})
    combined.update(scope="full", contributors=contributors, contributors_complete=True,
                    complete=False, reviewer="", acknowledged_review=False,
                    coordinator_confirmation_required=True)
    cross_links = [{"group": i, "related_group": related, "action": row["action"],
                    "related_action": combined["groups"][related]["action"]}
                   for i, row in combined["groups"].items() for related in row["related_groups"]
                   if owner_by_group[i] != owner_by_group[related]]
    combined["cross_part_links_to_review"] = cross_links
    destination.mkdir(parents=True)
    combined_path = destination / "merged_adjudication_draft.json"
    review.write_json(combined_path, combined)
    review.run(review.ROOT / "docs/review", review.ROOT / "docs/review/review_sample.json",
               review.ROOT / "docs/review/group_review_guidance.json", destination / "coordinator_review", combined_path)
    text = ["# 协调人最终确认", "", "两份分任务均已完成，原始统计保留。打开 coordinator_review/人工复核.html，检查关联组及统一口径，再填写协调人姓名并确认导出。", "",
            "## 跨份关联", "", "| 组 | 关联组 | 本组处理 | 关联组处理 |", "| --- | --- | --- | --- |"]
    text += [f"| {x['group']} | {x['related_group']} | {x['action']} | {x['related_action']} |" for x in cross_links]
    if not cross_links:
        text.append("| 无显式跨份关联 |  |  |  |")
    text += ["", "这一步不自动完成Task12主题冻结；G到机器模型的映射仍按原交接要求补齐。"]
    (destination / "协调人确认.md").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    distribute = sub.add_parser("split")
    distribute.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    distribute.add_argument("--output", type=Path, required=True)
    collect = sub.add_parser("merge")
    collect.add_argument("--assignments", type=Path, required=True)
    collect.add_argument("--part-a", type=Path, required=True)
    collect.add_argument("--part-b", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    if args.action == "split":
        data = split(args.review_dir, args.output)
        print(json.dumps([{k: p[k] for k in ("id", "high_priority_cases", "high_priority_groups", "estimated_effort", "review_file")}
                          for p in data["assignments"]], ensure_ascii=False, indent=2))
    else:
        data = merge(args.assignments, args.part_a, args.part_b, args.output, args.review_dir)
        print(json.dumps({"contributor_parts_complete": True, "coordinator_confirmation_required": True,
                          "cross_part_links": len(data["cross_part_links_to_review"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
