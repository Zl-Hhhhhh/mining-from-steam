"""Summarize unsupervised checks and prepare traceable sentence inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import analysis_common as common
import check_topic_stability as stability
from rework_unsupervised_topics import lexical_flags

ROOT = common.ROOT
BASE = ROOT / "output/topic_selection/task12-20261114"
REVIEW = ROOT / "output/topic_review/steam-review-20261107-v1"
GENERIC_REVIEW = re.compile(
    r"好玩|很好玩|非常好玩|不错|很不错|很棒|好评|差评|推荐|强烈推荐|不推荐|值得一玩|神作|垃圾游戏"
    r"|(?:(?:very|really) )?(?:good|great|nice|fun|awesome|amazing|excellent|bad|terrible)(?: game)?"
    r"|(?:highly )?recommend(?:ed| it)?|would recommend(?: it)?|not recommended|worth playing|worth it"
    r"|(?:[0-9](?:\.[0-9]+)?|10)\s*/\s*10|(?:[0-4](?:\.[0-9]+)?|5)\s*/\s*5"
)


def split_units(text):
    # This changes the modelling unit only; original reviews remain the RQ unit.
    pieces = re.split(r"[。！？!?；;\n\r]+|(?<=[A-Za-z0-9])\.(?=\s+[A-Z])|\bbut\b|但是|不过|然而|而且", text, flags=re.I)
    return [re.sub(r"\s+", " ", p).strip() for p in pieces if re.sub(r"\s+", " ", p).strip()]


def prepare_units(source, master, gate, *, allow_unmatched=False):
    units, review_map, generic_units = {}, {}, {}
    for row in source:
        review_id = row["recommendationid"]
        selected = review_map[review_id] = []
        if lexical_flags(row["text_clean"], gate)["template"]:
            continue
        for text in split_units(row["text_clean"]):
            generic = allow_unmatched and GENERIC_REVIEW.fullmatch(text.casefold().strip(" .!?。！？…"))
            flags = lexical_flags(text, gate)
            matched = bool(flags["matches"])
            if not generic and (not any(c.isalpha() for c in text) or not (matched or allow_unmatched)):
                continue

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            key = "S" + digest[:20]
            if generic:
                target = generic_units
                classification = {"auxiliary_route": "A01"}
            else:
                target = units
                classification = {
                    "lexical_categories_diagnostic": list(flags["matches"]),
                    "fit_candidate": matched and not flags["template"] and not flags["long_or_many_aspects"],
                }
                if key not in selected:
                    selected.append(key)
            item = target.setdefault(key, {
                "unit_id": key, "text": text, "text_sha256": digest, "review_ids": [], "appids": [],
                "source_languages": [], **classification,
            })
            if item["text"] != text:
                raise ValueError("Unit ID hash collision")
            for field, value in (("review_ids", review_id), ("appids", row["appid"]),
                                 ("source_languages", row["language"])):
                if value not in item[field]:
                    item[field].append(value)
    for row in master:
        review_map.setdefault(str(row["recommendationid"]), [])
    return units, review_map, generic_units


def run(destination, *, allow_unmatched=False):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new handoff directory")
    paths = [REVIEW / "source_mapping/group_source_mapping.json", REVIEW / "adjudicated-20261114/adjudication.json",
             BASE / "resampling_results.json", BASE / "unsupervised_rework/rework_results.json",
             BASE / "unsupervised_residual/rework_results.json", BASE / "nmf_rework/rework_results.json",
             ROOT / "config/topic_rework_residual.json", ROOT / "data/derived/embedding-bge/freeze_20261101-v1/model_reviews.jsonl",
             ROOT / "data/derived/embedding-bge/freeze_20261101-v1/master_index.jsonl", Path(__file__), ROOT / "config/embedding_bge.json"]
    hashes = {p.relative_to(ROOT).as_posix(): common.sha256(p) for p in paths}
    mapping, decisions, baseline, filtered, residual, nmf, gate = [common.read_json(p) for p in paths[:7]]
    actions = []
    for group in mapping["groups"]:
        d = decisions["groups"][group["group_id"]]
        actions.append({"group_id": group["group_id"], "candidate": group["run_id"], "source_topic": group["topic_id"],
            "action": d["action"], "name": d["final_answer"]["proposed_name"], "score": d["final_answer"]["coherence_rating"],
            "related_groups": d["related_groups"], "reason": d["rationale"],
            "implementation": "exclude_from_concrete_theme_catalog" if d["action"] == "generic" else
                              "align_semantic_name_only_across_independent_models" if d["action"] == "merge" else
                              "retain_as_candidate_and_recheck_outliers" if d["action"] == "retain" else
                              "do_not_transfer_whole_cluster_as_one_aspect; refit_on_content_units"})
    source = [json.loads(line) for line in paths[7].read_text(encoding="utf-8").splitlines()]
    master = [json.loads(line) for line in paths[8].read_text(encoding="utf-8").splitlines()]
    units, review_map, generic_units = prepare_units(source, master, gate, allow_unmatched=allow_unmatched)
    destination.mkdir(parents=True)
    stability.write(destination / "candidate_actions.json", {"status": "adjudication_applied_to_candidate_catalog",
        "rows": actions, "counts": dict(Counter(x["action"] for x in actions)), "task12_topics_frozen": False})
    unit_files = {"sentence_units.jsonl": units}
    if allow_unmatched:
        unit_files["auxiliary_units.jsonl"] = generic_units
    for filename, records in unit_files.items():
        with (destination / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(records):
                handle.write(json.dumps(records[key], ensure_ascii=False, allow_nan=False) + "\n")
    stability.write(destination / "review_to_units.json", review_map)
    embedding_config = common.read_json(ROOT / "config/embedding_bge.json")
    encoding_request = {"input": "sentence_units.jsonl", "input_sha256": common.sha256(destination / "sentence_units.jsonl"),
        "id_field": "unit_id", "text_field": "text", "encode_all_units": True,
        "ordered_unit_ids_sha256": hashlib.sha256(("\n".join(sorted(units)) + "\n").encode()).hexdigest(),
        "order_hash_format": "UTF-8 unit IDs in JSONL order, each followed by LF", "required_shape": [len(units),1024],
        "model_name": embedding_config["model_name"], "model_revision": embedding_config["model_revision"],
        "pooling": embedding_config["pooling"], "normalize_embeddings": True, "dtype": "float32",
        "max_seq_length": embedding_config["max_seq_length"], "cpu_threads":2,"max_batch_size":8,
        "output_files": ["sentence_vectors.npy", "unit_ids.json", "sentence_embedding_manifest.json"],
        "manifest_required_fields": ["model_revision","input_sha256","vectors_sha256","ordered_unit_ids_sha256","shape","pooling","normalize_embeddings"],
        "silent_truncation_allowed": False, "existing_whole_review_cache_applicable": False,
        "note": "This JSONL is a new unit schema, not a drop-in model_reviews.jsonl for the old fixed-row-count encoder. C should adapt the encoder input and preserve this ID order."}
    stability.write(destination / "encoding_request.json", encoding_request)
    manifest = {"status": "sentence_inputs_ready_encoding_not_run", "sources": hashes,
        "unit_count": len(units), "fit_candidate_units": sum(x["fit_candidate"] for x in units.values()),
        "master_review_count": len(master), "reviews_with_units": sum(bool(x) for x in review_map.values()),
        "human_labels_used_for_training": False, "recommendation_labels_used": False,
        "old_whole_review_vectors_reused_for_sentence_units": False, "task12_topics_frozen": False,
        "next_experiment": {"representation": "BGE-M3 same fixed revision on sentence texts", "cpu_threads": 2,
            "cluster_counts": [6,8,10], "seeds": [17,42], "subsampling_fraction": 0.8, "resamples": 5,
            "aggregation": "retain multiple unit topics per review; report no-evidence and ambiguous units separately"},
        "notes": ["Lexical rules screen inputs only, not final theme assignments; they can miss implicit aspects.",
            "No units does not establish that a review has no concrete aspect.",
            "Checkbox templates are withheld to avoid treating unselected options as opinions.",
            "Sentence source IDs and exact text are preserved; recommendation remains a review-level outcome."]}
    manifest["local_encoding_prerequisites"] = {"missing_packages": [p for p in ("torch", "sentence_transformers", "transformers") if importlib.util.find_spec(p) is None],
        "model_cache_present": (ROOT / embedding_config["model_cache_dir"]).exists()}
    if allow_unmatched:
        unmatched_ids = {key for key, unit in units.items() if not unit["lexical_categories_diagnostic"]}
        screening = {
            "policy": "allow_unmatched_application_units",
            "unmatched_application_units": len(unmatched_ids),
            "reviews_with_only_unmatched_units": sum(bool(keys) and unmatched_ids.issuperset(keys) for keys in review_map.values()),
            "generic_auxiliary_units": len(generic_units),
            "reviews_with_generic_auxiliary_units": len({rid for unit in generic_units.values() for rid in unit["review_ids"]}),
        }
        manifest["screening"] = screening
        manifest["notes"][0] = "Unmatched units are application-only candidates. Whole-fragment generic matches go to A01, outside the encoding input."
    output_names = [*unit_files, "review_to_units.json", "candidate_actions.json", "encoding_request.json"]
    manifest["outputs"] = [{"path": name, "sha256": common.sha256(destination / name)} for name in output_names]
    stability.write(destination / "manifest.json", manifest)

    lines = ["# Task12 检查结果", "", "结论：当前暂不冻结。30组意见已对应到模型处理表，已完成重采样及纯无监督重做对照。", "",
        "## 已完成", "", "- 原有18个候选，各做5次80%分层重采样，共90次。NMF同时重建词表和IDF。",
        "- 新增20个无监督候选：内容筛选BGE、去主方向BGE、内容筛选NMF；另做50次重采样。",
        "- 人工标签和推荐标签均未进入训练；计算顺序运行、2线程，原向量和历史结果保留。", "",
        "## 关键结果", "", "| 方案 | 平均重采样ARI | 范围 | 主要问题 |", "| --- | ---: | --- | --- |"]
    wanted = [("BGE原K=6",baseline,"bge/full_kmeans_k6_s17","稳定但方面混杂"),
        ("BGE原K=10",baseline,"bge/full_kmeans_k10_s17","泛评占据5/10组"),
        ("NMF原英文K=10",baseline,"nmf_aspect_v2/english_k10_s17","需拆分混杂与泛评组"),
        ("NMF原中文K=10",baseline,"nmf_aspect_v2/schinese_k10_s17","需拆分混杂与泛评组"),
        ("BGE内容筛选K=8",filtered,"filtered_kmeans_k8_s17","仍有综合好评／负评簇"),
        ("BGE去主方向K=8",residual,"residual_kmeans_k8_s17","部分方面更清楚，但稳定性下降"),
        ("NMF内容筛选英文K=6",nmf,"content_nmf_english_k6_s17","最大簇约71%，仍混杂"),
        ("NMF内容筛选中文K=8",nmf,"content_nmf_schinese_k8_s17","最大簇约59%，仍混杂")]
    for name, result, key, problem in wanted:
        summaries=result.get("summary",result.get("resampling_summary"))
        item=next(x for x in summaries if x["candidate"]==key)["ari"]
        lines.append(f"| {name} | {item['mean']:.3f} | {item['minimum']:.3f}–{item['maximum']:.3f} | {problem} |")
    lines += ["", "ARI衡量分组稳定性，不代表主题正确率。层次聚类只比较重叠样本，未外推到全量。", "",
        "## 裁决如何落实", "", "[30组处理表](candidate_actions.json)：泛评组退出具体主题目录；G10/G23只统一语义名称；混杂组转入重做。不同语言的NMF编号未直接合并。", "",
        "## 下一步", "", f"已准备 {len(units):,} 个去重句子／片段，其中 {manifest['fit_candidate_units']:,} 个可作拟合候选；覆盖 {manifest['reviews_with_units']:,}/{len(master):,} 条评论。",
        "使用[句子输入](sentence_units.jsonl)重新编码，再做无监督聚类；[原评论对应表](review_to_units.json)保留全部评论，供多主题汇总。旧整评向量不能冒充句子向量。",
        f"编码要求见[encoding_request.json](encoding_request.json)：固定原BGE修订，输出{len(units)}×1024向量、unit_ids.json及带输入／向量哈希的manifest。该输入采用新unit_id结构，需要适配编码入口。",
        "本机日常环境缺少torch、sentence-transformers及模型权重，句子编码尚未运行。",
        "C负责句子向量；B继续做聚类比较与稳定性；全员确认主题定义、反例及低置信度处理后，再冻结topics-v1。", "",
        "本轮没有生成正式主题比例或RQ2/RQ4结果；全量最终主题、权重与置信度须在有效主方案确定后输出。", ""]
    if allow_unmatched:
        lines.extend([
            "## 放宽词表筛选的候选", "",
            f"- 无词表命中的新增候选共 {screening['unmatched_application_units']:,} 个去重片段，全部仅供主题应用。",
            f"- {screening['reviews_with_only_unmatched_units']:,} 条评论只有无词表命中的候选。",
            f"- [A01 泛评片段](auxiliary_units.jsonl)共 {screening['generic_auxiliary_units']:,} 个去重片段，不进入编码。",
            "",
        ])
    (destination / "README.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    if any(common.sha256(ROOT/path)!=digest for path,digest in hashes.items()):raise RuntimeError("Source changed")
    fields = ("unit_count", "fit_candidate_units", "master_review_count", "reviews_with_units", "screening")
    print(json.dumps({key: manifest[key] for key in fields if key in manifest}, ensure_ascii=False))
    return manifest


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "handoff")
    parser.add_argument("--allow-unmatched", action="store_true", help="Retain unmatched text for topic application and export A01 generic fragments.")
    args = parser.parse_args()
    run(args.output, allow_unmatched=args.allow_unmatched)
