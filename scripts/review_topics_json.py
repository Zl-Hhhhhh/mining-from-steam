"""Compare Task11 multi-label JSON submissions without replacing human judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_FIELDS = ("observed_language", "aspect_presence", "aspect_labels", "reason",
               "expression_status", "expression_flags")
FIELD_NAMES = {"observed_language": "语言", "aspect_presence": "有无具体方面",
               "aspect_labels": "方面标签", "reason": "无方面／不确定原因",
               "expression_status": "表达状态", "expression_flags": "表达标记"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def answer_errors(answer, sample, stage):
    if not isinstance(answer, dict):
        return ["答案必须是对象"]
    errors = []
    if stage == "groups":
        if answer.get("group_kind") not in dict(sample["group_kinds"]):
            errors.append("组性质无效")
        if type(answer.get("coherence_rating")) is not int or answer["coherence_rating"] not in (0, 1, 2):
            errors.append("一致性评分无效")
        for key in ("proposed_name", "notes"):
            if not isinstance(answer.get(key), str) or not answer[key].strip():
                errors.append(f"{key} 缺失")
        return errors
    for key in ("observed_language", "aspect_presence", "reason", "expression_status",
                "style_evidence", "notes"):
        if not isinstance(answer.get(key), str):
            errors.append(f"{key} 必须是字符串")
    labels, flags, evidence = answer.get("aspect_labels"), answer.get("expression_flags"), answer.get("aspect_evidence")
    for key, values, allowed in (("aspect_labels", labels, sample["aspects"]),
                                 ("expression_flags", flags, sample["styles"])):
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            errors.append(f"{key} 必须是字符串列表")
        elif len(set(values)) != len(values) or any(x not in allowed for x in values):
            errors.append(f"{key} 有重复或无效值")
    if not isinstance(evidence, dict) or any(not isinstance(v, str) for v in evidence.values()):
        errors.append("方面证据格式错误")
    if errors:
        return errors
    presence, status = answer["aspect_presence"], answer["expression_status"]
    if answer["observed_language"] not in sample["languages"]:
        errors.append("实际语言无效")
    if presence not in ("yes", "no", "uncertain"):
        errors.append("有无具体方面无效")
    if presence == "yes" and not labels:
        errors.append("有方面但未选标签")
    if presence == "no" and labels:
        errors.append("无方面但仍有标签")
    if presence in ("no", "uncertain"):
        key = "no_reasons" if presence == "no" else "uncertain_reasons"
        if answer["reason"] not in dict(sample[key]):
            errors.append("原因无效")
    if presence == "uncertain" and not answer["notes"].strip():
        errors.append("缺少不确定说明")
    if "other_concrete" in labels and not answer["notes"].strip():
        errors.append("其他具体方面缺少说明")
    if any(not evidence.get(label, "").strip() for label in labels):
        errors.append("缺少所选方面的证据")
    if status not in ("ordinary", "special", "uncertain"):
        errors.append("表达状态无效")
    if status == "ordinary" and flags:
        errors.append("普通表达仍有特殊标记")
    if status == "special" and not flags:
        errors.append("特殊表达未选标记")
    if status in ("special", "uncertain") and not answer["style_evidence"].strip():
        errors.append("缺少表达证据或不确定说明")
    return errors


def validate_submissions(sample, submissions):
    checks = []
    for stage in ("cases", "groups"):
        ids = [item["id"] for item in sample[stage]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate sample IDs: {stage}")
        for rater in ("A", "B"):
            data = submissions[(rater, stage)]
            for key, expected in (("schema_version", "1.0"), ("package_id", sample["package_id"]),
                                  ("sample_sha256", sample["sample_sha256"]),
                                  ("rater_id", rater), ("stage", stage)):
                if data.get(key) != expected:
                    raise ValueError(f"{rater}/{stage}: {key} mismatch")
            if data.get("complete") is not True or data.get("acknowledged_codebook") is not True:
                raise ValueError(f"{rater}/{stage}: submission is not confirmed complete")
            if not isinstance(data.get("rater_name"), str) or not data["rater_name"].strip():
                raise ValueError(f"{rater}/{stage}: rater name missing")
            if stage == "groups" and data.get("stage1_import_confirmed") is not True:
                raise ValueError(f"{rater}/{stage}: stage1 gate not confirmed")
            answers = data.get("answers")
            if not isinstance(answers, dict) or set(answers) != set(ids):
                raise ValueError(f"{rater}/{stage}: answer IDs differ from sample")
            bad = {i: answer_errors(answers[i], sample, stage) for i in ids}
            bad = {i: errors for i, errors in bad.items() if errors}
            if bad:
                raise ValueError(f"{rater}/{stage}: invalid answers: {bad}")
            checks.append({"rater_id": rater, "stage": stage, "answer_count": len(ids),
                           "status": "passed", "ai_assistance": data.get("ai_assistance"),
                           "ai_assistance_declaration": "present" if data.get("ai_assistance") else "not_declared"})
    return checks


def agreement(pairs, weighted=False):
    """Cohen kappa with explicit denominators and null for a degenerate marginal."""
    n = len(pairs)
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    matrix = Counter(pairs)
    result = {"n": n, "matches": sum(a == b for a, b in pairs),
              "agreement": sum(a == b for a, b in pairs) / n if n else None,
              "A_counts": dict(left), "B_counts": dict(right),
              "confusion": [{"A": a, "B": b, "n": count} for (a, b), count in sorted(matrix.items())]}
    if not n:
        result.update(cohen_kappa=None, kappa_note="没有可比较记录")
        return result
    expected = sum(left[k] * right[k] for k in left.keys() | right.keys()) / n ** 2
    result["cohen_kappa"] = (result["agreement"] - expected) / (1 - expected) if expected < 1 - 1e-12 else None
    result["kappa_note"] = "" if result["cohen_kappa"] is not None else "所有记录为同一类别，κ 不定义"
    if weighted:
        observed_loss = sum((a - b) ** 2 for a, b in pairs) / n
        expected_loss = sum(count_a * count_b * (a - b) ** 2
                            for a, count_a in left.items() for b, count_b in right.items()) / n ** 2
        result["quadratic_weighted_kappa"] = 1 - observed_loss / expected_loss if expected_loss else None
    return result


def binary_metrics(pairs, excluded=0):
    result = agreement(pairs)
    n11 = sum(a == 1 and b == 1 for a, b in pairs)
    n00 = sum(a == 0 and b == 0 for a, b in pairs)
    positive_total = sum(a + b for a, b in pairs)
    negative_total = 2 * len(pairs) - positive_total
    result.update(excluded_uncertain=excluded,
                  positive_agreement=2 * n11 / positive_total if positive_total else None,
                  negative_agreement=2 * n00 / negative_total if negative_total else None)
    return result


def differences(a, b):
    fields = []
    for key in CORE_FIELDS:
        if key == "reason" and not (a["aspect_presence"] == b["aspect_presence"] and a["aspect_presence"] in ("no", "uncertain")):
            continue
        left, right = a[key], b[key]
        if key in ("aspect_labels", "expression_flags"):
            left, right = set(left), set(right)
        if left != right:
            fields.append(key)
    return fields


def make_case_records(sample, submissions):
    rows = []
    for case in sample["cases"]:
        i, text = case["id"], case["text"]
        a, b = (submissions[(r, "cases")]["answers"][i] for r in ("A", "B"))
        diff, checks = differences(a, b), []
        for rater, answer in (("A", a), ("B", b)):
            if answer["aspect_presence"] == "uncertain":
                checks.append(f"{rater}：方面不确定")
            if answer["expression_status"] == "uncertain":
                checks.append(f"{rater}：表达不确定")
            if answer["observed_language"] == "uncertain":
                checks.append(f"{rater}：语言无法判断")
            for label in answer["aspect_labels"]:
                if answer["aspect_evidence"][label] not in text:
                    checks.append(f"{rater}：{sample['aspects'][label]}证据不是连续原文")
        high = any(k in diff for k in ("aspect_presence", "aspect_labels")) or any("方面不确定" in s for s in checks)
        rows.append({**case, "A": a, "B": b, "differences": diff, "checks": checks,
                     "requires_review": bool(diff or checks), "priority": "high" if high else "normal"})
    return rows


def make_group_records(sample, submissions, guidance):
    rows = []
    for group in sample["groups"]:
        i = group["id"]
        a, b = (submissions[(r, "groups")]["answers"][i] for r in ("A", "B"))
        diff = [k for k in ("group_kind", "coherence_rating") if a[k] != b[k]]
        checks = []
        for rater, answer in (("A", a), ("B", b)):
            if answer["group_kind"] == "generic_evaluation" and answer["coherence_rating"] > 0:
                checks.append(f"{rater}：泛评组却给具体方面一致性 {answer['coherence_rating']} 分")
            for reference in re.findall(r"(?<![A-Za-z])E(\d+)", answer["notes"]):
                if not 1 <= int(reference) <= len(group["examples"]):
                    checks.append(f"{rater}：E{reference} 不在该组例子中")
        hint = guidance.get(i, {})
        if checks or hint.get("priority") == "high":
            priority = "high"
        else:
            priority = "normal"
        rows.append({**group, "A": a, "B": b, "differences": diff, "checks": checks,
                     "guidance": hint, "requires_review": True, "priority": priority})
    return rows


def case_metrics(rows, sample):
    result = {}
    for field in ("observed_language", "aspect_presence", "expression_status"):
        result[field] = agreement([(r["A"][field], r["B"][field]) for r in rows])
    certain = [r for r in rows if all(r[x]["aspect_presence"] != "uncertain" for x in ("A", "B"))]
    certain_styles = [r for r in rows if all(r[x]["expression_status"] != "uncertain" for x in ("A", "B"))]
    result["aspect_labels"] = {k: binary_metrics([(int(k in r["A"]["aspect_labels"]), int(k in r["B"]["aspect_labels"]))
                                                for r in certain], len(rows) - len(certain)) for k in sample["aspects"]}
    result["expression_flags"] = {k: binary_metrics([(int(k in r["A"]["expression_flags"]), int(k in r["B"]["expression_flags"]))
                                                   for r in certain_styles], len(rows) - len(certain_styles)) for k in sample["styles"]}
    result["aspect_set_exact"] = {"n": len(certain), "matches": sum(set(r["A"]["aspect_labels"]) == set(r["B"]["aspect_labels"]) for r in certain),
                                  "excluded_uncertain": len(rows) - len(certain)}
    n = result["aspect_set_exact"]["n"]
    result["aspect_set_exact"]["agreement"] = result["aspect_set_exact"]["matches"] / n if n else None
    jaccards = [len(set(r["A"]["aspect_labels"]) & set(r["B"]["aspect_labels"])) / len(set(r["A"]["aspect_labels"]) | set(r["B"]["aspect_labels"]))
                for r in certain if set(r["A"]["aspect_labels"]) | set(r["B"]["aspect_labels"])]
    result["aspect_jaccard_nonempty_union"] = {"n": len(jaccards), "mean": sum(jaccards) / len(jaccards) if jaccards else None,
                                               "note": "仅在双方均确定且至少一方选了方面的记录计算，空集对不计入"}
    return result


def analyse(sample, submissions, guidance):
    validation = validate_submissions(sample, submissions)
    cases, groups = make_case_records(sample, submissions), make_group_records(sample, submissions, guidance)
    metrics = {"cases": case_metrics(cases, sample),
               "groups": {field: agreement([(g["A"][field], g["B"][field]) for g in groups], weighted=field == "coherence_rating")
                          for field in ("group_kind", "coherence_rating")}}
    metrics["by_agreed_language"] = {}
    for language in ("english", "schinese", "mixed", "other", "uncertain"):
        rows = [r for r in cases if r["A"]["observed_language"] == r["B"]["observed_language"] == language]
        metrics["by_agreed_language"][language] = case_metrics(rows, sample)
    metrics["language_disagreement_n"] = sum(r["A"]["observed_language"] != r["B"]["observed_language"] for r in cases)
    counts = {field: sum(field in r["differences"] for r in cases) for field in CORE_FIELDS}
    counts.update(cases_total=len(cases), cases_requiring_review=sum(r["requires_review"] for r in cases),
                  cases_high_priority=sum(r["requires_review"] and r["priority"] == "high" for r in cases),
                  groups_total=len(groups), groups_kind_disagreement=sum("group_kind" in g["differences"] for g in groups),
                  groups_rating_disagreement=sum("coherence_rating" in g["differences"] for g in groups),
                  groups_high_priority=sum(g["priority"] == "high" for g in groups))
    return {"schema_version": "1.0", "package_id": sample["package_id"], "sample_sha256": sample["sample_sha256"],
            "status": "original_answers_compared_pending_human_adjudication", "validation": validation,
            "counts": counts, "metrics": metrics, "cases": cases, "groups": groups,
            "labels": {k: sample[k] for k in ("aspects", "styles", "languages", "no_reasons", "uncertain_reasons", "group_kinds")},
            "limitations": ["统计基于讨论前的原始答卷，裁决后不重算成独立一致性。",
                            "B 有 AI 预填后人工审查记录；A 未声明 AI 来源，不等于已验证完全无辅助。指标不称为双人完全无辅助标注信度。",
                            "方面或表达不确定的记录分别排除对应二元标签比较，并显式报告分母；不确定不折算为未选。",
                            "语言分层仅使用双方实际语言判断一致的记录；语言分歧单列，不按某一方语言静默归组。",
                            "组名不按文字是否相同计算一致率；双方相同判断也可能违背规则，30 组均需人工确认。",
                            "未提供匿名 G 编号到机器候选的协调人映射，不能据此给 NMF／BGE 排名或决定最终 K。",
                            "核验样本不是总体随机样本或独立留出准确率测试；Task12 的稳定性及主题冻结尚未完成。"]}


def input_fingerprint(files):
    stable = [{"role": f["role"], "sha256": f["sha256"]} for f in sorted(files, key=lambda x: x["role"])]
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decision_errors(data, result, sample):
    errors = []
    if not isinstance(data, dict):
        return ["裁决文件必须是对象"]
    if data.get("scope", "full") != "full":
        errors.append("分任务裁决请先使用 split_review_adjudication.py merge 合并")
    if type(data.get("complete")) is not bool:
        errors.append("完成状态必须是布尔值")
    for key in ("package_id", "sample_sha256", "source_fingerprint"):
        if data.get(key) != result.get(key):
            errors.append(f"{key} 不匹配")
    if data.get("schema_version") != "1.0" or data.get("stage") != "adjudication":
        errors.append("裁决文件格式无效")
    if not isinstance(data.get("cases"), dict) or not isinstance(data.get("groups"), dict):
        return errors + ["裁决记录必须是对象"]
    case_ids, group_ids = {r["id"] for r in result["cases"]}, {g["id"] for g in result["groups"]}
    if set(data["cases"]) - case_ids or set(data["groups"]) - group_ids:
        errors.append("包含未知裁决编号")
    for case_id, row in data["cases"].items():
        if not isinstance(row, dict):
            errors.append(f"{case_id} 格式错误")
            continue
        if row.get("status") == "reviewed":
            if not str(row.get("rationale", "")).strip():
                errors.append(f"{case_id} 缺少裁决理由")
            if row.get("basis") not in ("A", "B", "custom"):
                errors.append(f"{case_id} 裁决依据无效")
            elif row["basis"] in ("A", "B"):
                original = next(r for r in result["cases"] if r["id"] == case_id)[row["basis"]]
                if row.get("final_answer") != original:
                    errors.append(f"{case_id} 已修改原判断，请将裁决依据设为 custom")
            errors.extend(f"{case_id}: {e}" for e in answer_errors(row.get("final_answer"), sample, "cases"))
        elif row.get("status") != "pending":
            errors.append(f"{case_id} 状态无效")
    for group_id, row in data["groups"].items():
        if not isinstance(row, dict):
            errors.append(f"{group_id} 格式错误")
            continue
        if row.get("status") == "reviewed":
            if row.get("action") not in ("retain", "merge", "split", "rework", "generic", "uncertain"):
                errors.append(f"{group_id} 处理动作无效")
            if not str(row.get("rationale", "")).strip():
                errors.append(f"{group_id} 缺少处理理由")
            errors.extend(f"{group_id}: {e}" for e in answer_errors(row.get("final_answer"), sample, "groups"))
            related = row.get("related_groups", [])
            if not isinstance(related, list) or any(x not in group_ids or x == group_id for x in related):
                errors.append(f"{group_id} 关联组编号无效")
            if row.get("action") == "merge" and not related:
                errors.append(f"{group_id} 合并必须注明关联组")
        elif row.get("status") != "pending":
            errors.append(f"{group_id} 状态无效")
    if data.get("complete"):
        if not str(data.get("reviewer", "")).strip() or data.get("acknowledged_review") is not True:
            errors.append("缺少裁决人或复核确认")
        for r in result["cases"] + result["groups"]:
            stage = "cases" if r["id"] in case_ids else "groups"
            if r["requires_review"] and data[stage].get(r["id"], {}).get("status") != "reviewed":
                errors.append(f"{r['id']} 尚未复核")
    return errors


def make_template(result):
    return {"schema_version": "1.0", "stage": "adjudication", "package_id": result["package_id"],
            "sample_sha256": result["sample_sha256"], "source_fingerprint": result["source_fingerprint"],
            "complete": False, "reviewer": "", "acknowledged_review": False,
            "cases": {r["id"]: {"status": "pending", "basis": "", "final_answer": None, "rationale": ""}
                      for r in result["cases"] if r["requires_review"]},
            "groups": {g["id"]: {"status": "pending", "action": "", "related_groups": [], "final_answer": None, "rationale": ""}
                       for g in result["groups"]}}


def pct(value):
    return "不定义" if value is None else f"{value:.1%}"


def kappa(value):
    return "不定义" if value is None else f"{value:.3f}"


def write_summary(result, output):
    c, m = result["counts"], result["metrics"]
    progress = "本轮人工裁决已导入；Task12 主题冻结尚未完成。" if result["human_adjudication_complete"] else "人工裁决和 Task12 主题冻结尚未完成。"
    if result.get("adjudication", {}).get("adjudicator_type") == "ai":
        progress = "已导入 AI 裁决；这不代表新增人工签署或 Task12 主题冻结。"
    lines = ["# Task11 双人核验对照", "", "原始四份答卷已验证，以下一致性来自讨论前判断。" + progress, "",
             f"人工复核入口：[人工复核.html](人工复核.html)。共 {c['cases_requiring_review']} 条评论进入复核，其中 {c['cases_high_priority']} 条涉及方面分歧或不确定；另有 30 组候选命名全部待确认。", "",
             "## 一致性", "", "| 判断 | 一致率 | Cohen κ | 分母 |", "| --- | ---: | ---: | ---: |"]
    for field in ("observed_language", "aspect_presence", "expression_status"):
        a = m["cases"][field]
        lines.append(f"| {FIELD_NAMES[field]} | {pct(a['agreement'])} | {kappa(a['cohen_kappa'])} | {a['n']} |")
    for field, name in (("group_kind", "候选组性质"), ("coherence_rating", "候选组评分")):
        a = m["groups"][field]
        lines.append(f"| {name} | {pct(a['agreement'])} | {kappa(a['cohen_kappa'])} | {a['n']} |")
    lines += ["", "| 方面 | 一致率 | Cohen κ | 双方阳性一致率 | 分母 |", "| --- | ---: | ---: | ---: | ---: |"]
    for field, a in m["cases"]["aspect_labels"].items():
        lines.append(f"| {result['labels']['aspects'][field]} | {pct(a['agreement'])} | {kappa(a['cohen_kappa'])} | {pct(a['positive_agreement'])} | {a['n']} |")
    lines += ["", f"方面标签比较排除了任一方方面不确定的 {next(iter(m['cases']['aspect_labels'].values()))['excluded_uncertain']} 条；阳性一致率帮助识别大量共同未选导致的表面高一致。", "",
              "## 你需要人工核对的内容", "",
              f"- 有无具体方面不同：{c['aspect_presence']} 条；方面标签不同：{c['aspect_labels']} 条。两项有重叠。",
              f"- 语言不同：{c['observed_language']} 条；表达状态不同：{c['expression_status']} 条；表达标记不同：{c['expression_flags']} 条。",
              f"- 30 组中，性质不同 {c['groups_kind_disagreement']} 组，评分不同 {c['groups_rating_disagreement']} 组；{c['groups_high_priority']} 组被列为优先复核。",
              *[f"- {r['id']}：{check}。" for r in result["cases"] for check in r["checks"] if "不是连续原文" in check],
              "- 各组页面中的 AI 提示只是待讨论意见，首次生成时裁决字段全部留空；填写自己的最终判断和简短理由。", "",
              "| 组 | 优先讨论的问题 |", "| --- | --- |"]
    for group in result["groups"]:
        if group["priority"] == "high":
            issue = group["guidance"].get("question", "；".join(group["checks"]))
            lines.append(f"| {group['id']} | {issue} |")
    lines += ["", "## 保存和后续", "",
              "在离线页填写裁决人，逐项选择或修改最终标签、填写理由，点击“确认此项并下一项”。每次离开前导出 JSON 备份；换电脑或移动文件后可导入备份。完成导出会检查所有复核项，但允许最终判断仍为不确定。", "",
              "四份原始答卷不改写，裁决另存；页面不把 AI 提示自动写成最终结论。导出的裁决可通过脚本的 --decisions 参数验证并汇入新的输出目录。", "",
              "还需协调人提供 G 编号到机器候选的映射，然后形成对应模型的合并、拆分或重做清单。未提供映射时只讨论匿名 G 组，不猜模型来源。之后由全员进行 Task12 稳定性检查和主题冻结。", "",
              "## 方法与来源限制", ""] + [f"- {x}" for x in result["limitations"]]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run(input_dir, sample_path, guidance_path, output, decisions_path=None):
    input_dir, sample_path, guidance_path, output = map(Path, (input_dir, sample_path, guidance_path, output))
    if output.exists():
        raise FileExistsError("Use a new output directory; existing review evidence is never overwritten")
    sample, guidance = read_json(sample_path), read_json(guidance_path)
    if set(guidance) != {g["id"] for g in sample["groups"]}:
        raise ValueError("Group guidance IDs differ from sample")
    submissions, files = {}, []
    for rater in ("A", "B"):
        for stage in ("cases", "groups"):
            path = input_dir / f"{sample['package_id']}_{rater}_{stage}_complete.json"
            submissions[(rater, stage)] = read_json(path)
            files.append({"role": f"{rater}_{stage}", "filename": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
    files.extend({"role": role, "filename": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
                 for role, path in (("sample", sample_path), ("ai_group_guidance", guidance_path),
                                    ("codebook", input_dir / "CODEBOOK.md")))
    result = analyse(sample, submissions, guidance)
    result.update(generated_at=datetime.now(timezone.utc).isoformat(), sources=files,
                  source_fingerprint=input_fingerprint(files), generator_sha256=sha(Path(__file__)),
                  original_answers_unchanged=True, human_adjudication_complete=False, task12_topics_frozen=False)
    template = make_template(result)
    if decisions_path:
        decisions = read_json(decisions_path)
        errors = decision_errors(decisions, result, sample)
        if errors:
            raise ValueError("Invalid adjudication: " + "; ".join(errors))
        template = decisions
        result["adjudication"] = {"filename": Path(decisions_path).name, "sha256": sha(decisions_path),
                                   "complete": bool(decisions.get("complete")), "task12_topics_frozen": False}
        actor = decisions.get("adjudicator_type", "human")
        result["adjudication"]["adjudicator_type"] = actor
        result["human_adjudication_complete"] = bool(decisions.get("complete")) and actor != "ai"
        result["review_session_id"] = sha(decisions_path)
        if actor == "ai":
            result["decision_context"] = "已载入 AI 裁决，未代替复核者签署；Task12 尚未冻结。"
        result["status"] = "adjudication_imported_not_team_topic_freeze"
    result["initial_decisions"] = template
    result["report_template_sha256"] = sha(Path(__file__).with_suffix(".html"))
    output.mkdir(parents=True)
    write_json(output / "review_results.json", result)
    write_json(output / "adjudication_template.json", template)
    write_summary(result, output)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    html = Path(__file__).with_suffix(".html").read_text(encoding="utf-8").replace("__REVIEW_PAYLOAD__", payload)
    (output / "人工复核.html").write_text(html, encoding="utf-8", newline="\n")
    for record in files[:4]:
        if sha(input_dir / record["filename"]) != record["sha256"]:
            raise RuntimeError("An original submission changed during analysis")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "docs/review")
    parser.add_argument("--sample", type=Path, default=ROOT / "docs/review/review_sample.json")
    parser.add_argument("--guidance", type=Path, default=ROOT / "docs/review/group_review_guidance.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    result = run(args.input_dir, args.sample, args.guidance, args.output, args.decisions)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
