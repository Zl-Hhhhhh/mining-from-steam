"""Reconstruct anonymous review group anchors from archived candidate evidence."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import review_topics_json as review

ROOT = review.ROOT
NMF_ROOTS = (ROOT / "output/baseline_nmf/freeze_20261101",
             ROOT / "output/baseline_nmf/freeze_20261101_aspect_v2")
BGE_ROOT = ROOT / "output/clustering-bge/freeze_20261101-v1"
SAMPLE = ROOT / "docs/review/review_sample.json"
MASTER = ROOT / "data/cleaned/freeze_20261101/clean_master.csv"


def normalized(text):
    """Only normalize whitespace: no translation, fuzzy matching, or case folding."""
    return re.sub(r"\s+", " ", text).strip()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def compatible_topics(group, text_ids, assignments):
    sets = [{assignments[i] for i in text_ids[normalized(example["text"])]
             if i in assignments and assignments[i] >= 0} for example in group["examples"]]
    return set.intersection(*sets) if sets else set()


def choose_anchor(groups, candidates, compatible, centers, family):
    eligible = []
    for candidate in candidates:
        if candidate["family"] != family:
            continue
        topics = []
        for group in groups:
            options = compatible[group["id"]].get(candidate["run_id"], set()).copy()
            if family == "bge_shared":
                options = {topic for topic in options if all(
                    normalized(e["text"]) in centers[(candidate["candidate"], topic)]
                    for e in group["examples"] if e["kind"] == "center")}
            if len(options) != 1:
                break
            topics.append(next(iter(options)))
        if len(topics) == len(groups) and len(set(topics)) == len(groups):
            eligible.append((candidate, dict(zip((g["id"] for g in groups), topics))))
    if len(eligible) != 1:
        raise ValueError(f"{family}: expected one archive-compatible anchor, found {len(eligible)}")
    return eligible[0]


def recover(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new source mapping directory")
    sample = review.read_json(SAMPLE)
    texts = {normalized(e["text"]) for g in sample["groups"] for e in g["examples"]}
    text_ids, metadata = defaultdict(set), {}
    paths = [SAMPLE, MASTER, BGE_ROOT / "representative_reviews.csv"]
    for row in csv_rows(MASTER):
        for field in ("review", "review_clean"):
            value = normalized(row[field])
            if value in texts:
                text_ids[value].add(row["recommendationid"])
                metadata[row["recommendationid"]] = {k: row[k] for k in ("appid", "language")}
    if any(not text_ids[t] for t in texts):
        raise ValueError("Some example texts do not match the archived master")
    wanted_ids = set.union(*text_ids.values())
    candidates = []
    for parent in (*NMF_ROOTS, BGE_ROOT):
        for path in sorted(parent.glob("*/assignments.csv")):
            bge = parent == BGE_ROOT
            match = re.search(r"_k(\d+)(?:_s(\d+))?$", path.parent.name)
            if not match:
                raise ValueError(f"Unrecognized candidate: {path}")
            candidate = {"run_id": path.parent.relative_to(ROOT).as_posix(), "candidate": path.parent.name,
                         "family": "bge_shared" if bge else "nmf_" + path.parent.name.split("_k")[0],
                         "source_language": "shared_multilingual" if bge else path.parent.name.split("_k")[0],
                         "model": "BGE-M3 + " + ("KMeans" if "kmeans" in path.parent.name else path.parent.name.split("_")[0]) if bge else "TF-IDF + NMF",
                         "k": int(match[1]), "seed": int(match[2]) if match[2] else None,
                         "assignment_file": path.relative_to(ROOT).as_posix(), "assignments": {}}
            for row in csv_rows(path):
                if row["recommendationid"] in wanted_ids:
                    # NMF's no_clear_theme is a provisional rejection. topic_index
                    # still records the raw model argmax used to build the examples.
                    topic = int(row["raw_cluster" if bge else "topic_index"])
                    candidate["assignments"][row["recommendationid"]] = topic
            candidates.append(candidate)
            paths.append(path)
    centers = defaultdict(set)
    for row in csv_rows(BGE_ROOT / "representative_reviews.csv"):
        if row["selection"].startswith("center"):
            centers[row["candidate"], int(row["cluster"])].add(normalized(row["text"]))
    compatible, by_family = {}, defaultdict(list)
    for group in sample["groups"]:
        matches = {candidate["run_id"]: compatible_topics(group, text_ids, candidate["assignments"])
                   for candidate in candidates}
        compatible[group["id"]] = {run: topics for run, topics in matches.items() if topics}
        families = {c["family"] for c in candidates if matches[c["run_id"]]}
        if len(families) != 1:
            raise ValueError(f"{group['id']}: ambiguous model family {families}")
        by_family[next(iter(families))].append(group)
    reconstructed, anchors = [], []
    for family, groups in sorted(by_family.items()):
        candidate, topics = choose_anchor(groups, candidates, compatible, centers, family)
        anchor = {k: v for k, v in candidate.items() if k != "assignments"}
        anchors.append(anchor)
        for group in groups:
            topic = topics[group["id"]]
            evidence = []
            for example in group["examples"]:
                all_ids = text_ids[normalized(example["text"])]
                valid = sorted(i for i in all_ids if candidate["assignments"].get(i) == topic)
                evidence.append({"example_id": example["id"], "kind": example["kind"],
                    "matching_review_ids_in_master": sorted(all_ids), "matching_ids_in_anchor_topic": valid,
                    "source_id_unique": len(valid) == 1,
                    "source_languages": sorted({metadata[i]["language"] for i in valid}),
                    "published_bge_center_match": normalized(example["text"]) in centers[(candidate["candidate"], topic)]
                                                  if family == "bge_shared" and example["kind"] == "center" else None})
            reconstructed.append({"group_id": group["id"], **anchor, "topic_index": topic,
                "topic_id": str(topic) if family == "bge_shared" else f"{candidate['source_language']}:T{topic:02}",
                "status": "reconstructed_from_archived_evidence", "examples": evidence,
                "other_six_example_compatible_runs": [
                    {"run_id": run, "topic_indices": sorted(options)}
                    for run, options in compatible[group["id"]].items() if run != candidate["run_id"]]})
    reconstructed.sort(key=lambda row: row["group_id"])
    result = {"schema_version": "1.0", "package_id": sample["package_id"], "generated_at": datetime.now(timezone.utc).isoformat(),
        "original_coordinator_key_found": False, "status": "all_30_group_anchors_reconstructed",
        "method": "Exact text after whitespace normalization; all six examples must share one raw topic; one archived run per model/language family and distinct topics across its groups; BGE center examples must also occur in published center or center_language rows.",
        "limitations": ["This is a reconstruction within available archived candidates, not the original anonymization manifest.",
            "A single group can match multiple runs; joint family coverage and center evidence disambiguate the anchor.",
            "Repeated identical texts may leave the exact recommendation ID ambiguous; all compatible IDs are retained.",
            "Source language is the dataset language field, not a new human actual-language annotation.",
            "Raw NMF topic_index does not override provisional no_clear_theme or create a final theme.",
            "Recovered k=10/seed=17 are review anchors, not a final model choice or topic freeze."],
        "generator_sha256": review.sha(Path(__file__)),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": review.sha(p)} for p in paths],
        "anchors": anchors, "groups": reconstructed, "task12_topics_frozen": False}
    destination.mkdir(parents=True)
    review.write_json(destination / "group_source_mapping.json", result)
    lines = ["# G01–G30 来源重建", "", "仓库没有新版原始协调人对照表；现已用例子原文、逐条归属及 BGE 中心记录重建全部30组的候选来源。", "",
             "[逐例证据与来源哈希](group_source_mapping.json)。这份结果属于归档证据重建，K=10、seed=17 仍是核验锚点。", "",
             "| G编号 | 模型 | 数据语言范围 | K | seed | 原主题／簇编号 |", "| --- | --- | --- | ---: | ---: | --- |"]
    languages = {"schinese": "中文", "english": "英文", "shared_multilingual": "中英共享"}
    for group in reconstructed:
        lines.append(f"| {group['group_id']} | {group['model']} | {languages[group['source_language']]} | {group['k']} | {group['seed']} | {group['topic_id']} |")
    lines += ["", "主题与簇编号从0开始。NMF来自 `output/baseline_nmf/freeze_20261101_aspect_v2/`，BGE来自 `output/clustering-bge/freeze_20261101-v1/full_kmeans_k10_s17/`。", "",
              "## 核对口径", "", "- 180个例子对应179条不同原文，均可在清洗主表中按原文匹配；只统一空白字符，没有模糊搜索或语义猜测。",
              "- 比较仓库两个NMF版本和全部BGE候选。每个模型／语言家族同时满足六例同主题、各组主题不同的条件，BGE还需匹配发布的中心例子，最终各剩一个锚点。",
              "- `no_clear_theme` 是NMF暂定拒绝结果，回溯使用其原始 `topic_index`；BGE负编号视为不适用，不当作簇。",
              "- 同文多条评论保留所有候选ID；这不影响已核实的组级主题归属，但没有恢复不存在的原始抽样ID清单。",
              "- 源语言标签不代替实际语言核验；此表不选择主模型，不冻结最终主题。", ""]
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.output)
    print(json.dumps({"groups": len(result["groups"]), "anchors": [a["run_id"] for a in result["anchors"]]}, indent=2))
