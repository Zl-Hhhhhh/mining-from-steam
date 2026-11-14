"""Apply the documented Task11 adjudication to the two submitted review drafts.

The drafts remain evidence, including their pending flags. Completion here means
AI adjudication under the user's instruction, not additional human signatures.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import review_topics_json as review
import split_review_adjudication as split

BASE = review.ROOT / "output/topic_review/steam-review-20261107-v1"
EXPECTED_DRAFT_SHA256 = (
    "86c4432b7d6289991d2f5f7ad9832f31ed3171c5aef95c50f70a4551d4de0c29",
    "6bd8af4202829761477b94ad227a5f8399aa5f73041f27ef9f0fcaca20690e26",
)


def resolve_cases(rows, originals):
    reasons = {}

    def patch(i, why, **fields):
        rows[i]["final_answer"].update(fields)
        reasons[i] = why

    def aspects(i, why, evidence, **fields):
        patch(i, why, aspect_presence="yes", aspect_labels=list(evidence),
              aspect_evidence=evidence, reason="", **fields)

    def no_aspect(i, why, reason="generic_evaluation", **fields):
        patch(i, why, aspect_presence="no", aspect_labels=[], aspect_evidence={},
              reason=reason, **fields)

    def adopt(i, who, why):
        rows[i]["final_answer"] = copy.deepcopy(originals[i][who])
        reasons[i] = why

    # Explicit aspects take precedence over a blanket 'generic praise' decision.
    adopt("R007", "B", "原文明示购买点数与直升机的付费建议，不是纯梗。")
    aspects("R009", "拍图腾、拍鬼照是可指出的游戏行为，玩笑不排除玩法方面。",
            {"gameplay": "低头拍图腾，裆下显形拍了个鬼照"})
    for i, phrase in {"R011": "good with friends", "R015": "跟朋友一起爽玩",
                      "R081": "跟朋友一起玩硬生生把恐怖游戏玩成了搞笑游戏",
                      "R211": "和朋友一起玩很好玩", "R272": "very fun play with friends"}.items():
        aspects(i, "明确把朋友同玩作为体验来源，统一归联机方面。", {"multiplayer_community": phrase})
    aspects("R016", "明确批评合作体验；保留复核者识别的反讽线索。",
            {"multiplayer_community": "Worst co op xp i had in 30years of playing games."})
    patch("R018", "保留人质相关玩法判断，将拼接词改为连续原文证据。",
          aspect_evidence={"gameplay": "我没看到有人质,不是只有没持枪的歹徒嘛"})
    for i, char in {"R028": "l", "R156": "a"}.items():
        no_aspect(i, "孤立字母不能确定语言或与游戏无关；按低信息符号记录。", "symbols_only",
                  observed_language="uncertain", notes="仅有孤立字母。",
                  style_evidence=f"仅有 {char}，表达含义不明。")
    for i in ("R030", "R057", "R250"):
        patch(i, "仅回忆过去，没有具体游戏方面；不归为与游戏无关。",
              reason="other_no_specific_aspect", notes="仅表达回忆。")
    patch("R034", "备注已指出可能借土豆指服务器，语境不足时保留不确定。",
          aspect_presence="uncertain", reason="missing_context", notes="土豆所指不明。",
          style_evidence="土豆发芽没，没发芽我去炸薯条了")
    aspects("R053", "连接超时是运行故障，同时发生在朋友联机中。",
            {"performance_stability": "connection timed out", "multiplayer_community": "Me and my friends cant play it 95% of the time"})
    aspects("R063", "开船导致眩晕属于身体不适，原文没有具体船只机制评价。",
            {"other_concrete": "就是那个开船老晕了"}, notes="其他具体方面：眩晕与身体舒适度。")
    no_aspect("R066", "仅称某类型游戏最好，未评价具体射击机制。")
    patch("R070", "PLATFORMING 是平台跳跃，不是多人联机；保留其他有证据的方面。",
          aspect_labels=[x for x in rows["R070"]["final_answer"]["aspect_labels"] if x != "multiplayer_community"])
    adopt("R076", "B", "明确说适合策略游戏入门，并评价扩展内容的必要性。")
    aspects("R092", "明确偏好开放世界形式，按探索玩法记录。", {"gameplay": "就乐意玩点开放世界"})
    patch("R097", "除咒骂外，原文也直接攻击设计者。",
          expression_flags=["profanity", "directed_attack"],
          style_evidence="cnm的劳资玩nm；生出你这个杂种玩意做出这种sb设计")
    patch("R101", "清除无关的感叹备注，保留本地化方面说明。", notes="其他具体方面：中文本地化/翻译。")
    aspects("R108", "除内容量和价格外，同图同敌人的反复战斗明确涉及重复玩法。",
            {"content_support": "10小时就可以体验完游戏的所有内容", "gameplay": "之后都是在同一张地图打同一个敌人",
             "price_monetization": "90元的价格完全不值"})
    aspects("R114", "更新后无法进入游戏，按规则归运行故障，不因更新二字加维护。",
            {"performance_stability": "更新后连游戏都进不去"})
    adopt("R122", "B", "grind 明确描述重复刷取的玩法，不是只有好玩。")
    patch("R124", "lol 是文字缩写，没有可见 emoji；保留难度判断。",
          expression_status="ordinary", expression_flags=[], style_evidence="")
    patch("R129", "枪打不到人支持操作判断，但不足以单独推定反讽。",
          expression_status="ordinary", expression_flags=[], style_evidence="")
    aspects("R130", "gameplay 不等于操作难度；按同类短评口径保留明确的剧情评价。",
            {"story_characters": "story!"})
    adopt("R137", "B", "原文评价特定战斗评级体验，且有可见的赞踩符号。")
    aspects("R152", "掉线归运行故障，封禁归联机管理，两种诉求并存。",
            {"performance_stability": "天天让我朋友掉线", "multiplayer_community": "还封禁她们"})
    aspects("R158", "反复催更仍有明确的更新维护诉求。", {"content_support": "快点更新阿"})
    aspects("R165", "更新不了描述技术故障，不是更新频率或内容的评价。",
            {"performance_stability": "更新不了"})
    no_aspect("R171", "仅有数字，无法核实特定梗或语言。", "symbols_only",
              expression_status="uncertain", expression_flags=[], style_evidence="9178 的含义不明。")
    patch("R174", "NO GO 在此表达否定推荐，没有必须保留的表达歧义。",
          expression_status="ordinary", expression_flags=[], style_evidence="")
    aspects("R182", "hard mechanics 评价机制难度，不能把难度证据记成玩法类型。",
            {"controls_difficulty": "hard game, hard mechanics", "multiplayer_community": "hard community"})
    adopt("R186", "B", "怀念的是早期版本的殖民地机制，具有具体对象。")
    adopt("R187", "B", "Disable parries 明确要求取消招架，属于操作与战斗平衡。")
    no_aspect("R188", "可玩性高和有意思仍属泛赞，与同类可玩性短评统一。")
    patch("R189", "笼统玩法打分未指具体机制；保留明确的剧情和重玩性。",
          aspect_labels=["story_characters", "content_support"])
    patch("R198", "沉浸式策略体验并非画面或声音证据，移除视听标签。",
          aspect_labels=["gameplay", "controls_difficulty", "content_support"])
    no_aspect("R202", "long hours 只给时长，不能直接推断重玩性或内容量。",
              notes="仅有总体评价和时长。")
    patch("R203", "孤立数字不确定语言，与其他纯数字记录统一。", observed_language="uncertain")
    patch("R208", "用构筑和反复重开的原文支撑玩法，并补回明示的咒骂符号与引用。",
          aspect_evidence={"gameplay": "Trying to play different builds each time is fun."},
          expression_status="special", expression_flags=["profanity", "meme_quote"],
          style_evidence="*#$%^*&#%；引用：I'd rather staple my nuts to the floor and jump")
    patch("R214", "字母标识不构成英文语句，飞马是文字而非 emoji。",
          observed_language="schinese", expression_flags=["profanity", "meme_quote"],
          style_evidence="飞马；bvvd")
    aspects("R215", "误封诉求归反作弊与联机管理，不因更新二字归维护。",
            {"multiplayer_community": "VacNet更新后乱jb封，已经第三次31天黄信了"}, observed_language="schinese")
    patch("R236", "单独妈妈缺少指代，不能确定与游戏无关或是特定梗。",
          aspect_presence="uncertain", reason="missing_context", notes="妈妈所指不明。",
          expression_status="uncertain", expression_flags=[], style_evidence="仅有妈妈，语境不明。")
    patch("R240", "时长是游玩记录，未直接说明内容量或重玩性；保留画面评价。",
          aspect_labels=["audiovisual"])
    aspects("R262", "对刷怪和被人带着玩的态度均有明确对象。",
            {"gameplay": "对刷怪游戏并不是很感冒", "multiplayer_community": "有老登带着玩很开心"})
    patch("R281", "疲惫感缺少具体原因，不足以判定与游戏无关。", reason="other_no_specific_aspect")
    aspects("R288", "联机卡死归性能与联机；没有安装 MOD 不是 MOD 内容评价。",
            {"performance_stability": "it would keep freezing", "multiplayer_community": "when my friend tried to join me"})
    aspects("R291", "除音乐外，还明确提到持续眩晕控制效果。",
            {"audiovisual": "great music", "controls_difficulty": "perma stunning you"})

    # Inactive evidence/reasons left by checkbox edits are not new annotations.
    for i, row in rows.items():
        a = row["final_answer"]
        a["aspect_evidence"] = {k: a["aspect_evidence"][k] for k in a["aspect_labels"]}
        if a["aspect_presence"] == "yes":
            a["reason"] = ""
        before = originals[i]
        row["basis"] = next((r for r in ("A", "B") if a == before[r]), "custom")
        row["status"] = "reviewed"
        row["rationale"] = reasons.get(i) or row["rationale"].strip() or "采纳分工复核选择；所选判断与原文、规则相容。"
    return reasons


# Score the current six examples, not a hypothetical cleaned or merged cluster.
# Relationships below describe semantic targets, not edits to anonymous model IDs.
GROUPS = [
    ("G01", "rework", "mixed_uninterpretable", 0, "混杂短评与联机故障", [], "E1、E3、E5 为泛评或梗；E4 有视听和游玩经历；E6 是联机故障及惩罚。没有共同方面，按原文重新分组。"),
    ("G02", "split", "generic_evaluation", 0, "可玩性泛评", [], "E1—E4 只有可玩性评价；E5 单列匹配公平，E6 单列挂钩机制。泛评主体不设玩法主题。"),
    ("G03", "retain", "single_aspect", 1, "开放世界与自由探索", [], "E1、E2、E6 支持开放世界，E3 仅提类型；E4、E5 不作支持例，清理后保留候选。"),
    ("G04", "retain", "single_aspect", 2, "服务器与运行故障", [], "E1—E3、E6 支持服务器问题；E5 涉及下载与优化。E4 夹杂政治内容，不用于推断具体故障。"),
    ("G05", "generic", "generic_evaluation", 0, "最佳游戏泛赞", ["G13", "G16", "G26", "G30"], "六例均为整体认可，没有共同具体方面；归泛赞参考类。"),
    ("G06", "split", "multiple_aspects", 1, "入门、联机、MOD与综合体验", [], "E1、E5 谈上手；E2、E3 谈朋友同玩；E2、E4 谈 MOD；E6 涉及操作、视听、剧情和性能。不能整体当泛赞，按方面拆分。"),
    ("G07", "retain", "single_aspect", 1, "无法游玩与运行故障", [], "E1、E4—E6 支持无法游玩；E2 只明确画面模糊，E3 是卡牌构筑。保留故障核心，视听与玩法例子另分。"),
    ("G08", "rework", "mixed_uninterpretable", 0, "梗、泛评与具体问题混合", [], "E2 为版本泛评，E1、E3 语境有限；E4 明确技术闪烁，E5 谈重玩和重复性，E6 涉及能力门槛。不能一律归泛评。"),
    ("G09", "generic", "generic_evaluation", 0, "负面泛评", [], "E1、E2、E4—E6 为笼统负评；E3 的玩家评价单列社区。泛评主体具体方面一致性为 0。"),
    ("G10", "merge", "single_aspect", 2, "朋友合作与联机体验", ["G23"], "E1—E3、E6 支持朋友同玩；E4 谈合作与设置，E5 同时肯定朋友游玩和跑酷。与 G23 统一语义主题，保留多标签。"),
    ("G11", "generic", "generic_evaluation", 0, "怀旧与童年回忆", [], "六例共同点是怀旧动机；E4 提卡牌肉鸽，但没有跨例共同具体玩法。保留怀旧描述，不作为具体方面主题。"),
    ("G12", "rework", "mixed_uninterpretable", 0, "比较性泛评与综合体验", [], "E1—E4 为比较或整体感受；E5 谈舒缓体验，E6 涉及画面、难度和 DLC。比较用词不能构成共同方面。"),
    ("G13", "generic", "generic_evaluation", 0, "整体认可与值得玩", ["G05", "G16", "G26", "G30"], "大部分仅说好玩或值得；E1 的 so much to do 可另记内容量。主体按泛赞处理，评分不按情绪一致性给 2。"),
    ("G14", "rework", "mixed_uninterpretable", 0, "低信息短句与混杂表达", [], "E1 提 MOD，E2、E3 语境或语言不清，E4—E6 是零散感受；没有可确认的共同具体方面。"),
    ("G15", "split", "multiple_aspects", 1, "多方面综合好评", [], "剧情、操作、玩法、视听等多方面并存；按方面保留多标签，不把共同好评当成单一研究主题。"),
    ("G16", "generic", "generic_evaluation", 0, "简短泛赞", ["G05", "G13", "G26", "G30"], "除 E3 明确杀鲨鱼的游戏行为外，其余是泛赞；E3 单列玩法，主体归泛赞。"),
    ("G17", "split", "multiple_aspects", 1, "操作、难度与上手门槛", ["G21"], "E1—E5 有操作、学习或战斗难度；E2、E4 另含性能问题，E6 是跑图和任务重复。难度部分与 G21 对齐，其余按方面拆分。"),
    ("G18", "rework", "mixed_uninterpretable", 0, "游玩时长与泛评", ["G24"], "E1—E4 主要为时长与泛评，不直接证明内容量或重玩性；E5 是学习门槛，E6 是全成就后的类型评价。按实际方面重新分组。"),
    ("G19", "rework", "mixed_uninterpretable", 0, "低信息感受与联机故障", [], "前几例多为泛评或语境不明；E5 明确联机卡死，应归故障；E6 为生活感受。不能整体当成一个主题。"),
    ("G20", "rework", "mixed_uninterpretable", 0, "厂商、版本与泛评混合", [], "E1、E2 针对厂商或续作，E3 比较版本，E6 谈重制升级；E4、E5 是泛赞。按各自含义重新分组。"),
    ("G21", "retain", "single_aspect", 2, "战斗与通关难度", ["G17"], "E1—E3、E6 明确评价难度；E4 叙事和过场、E5 语境不明，不作支持例。保留候选并对齐 G17 的难度部分。"),
    ("G22", "split", "multiple_aspects", 1, "多方面综合评论", ["G27"], "E1—E3 是多方面长评，但并非全有评分模板；E4 是泛评，E5 有剧情和合作体验，E6 有重玩性。按方面拆分，篇幅仅作文本特征。"),
    ("G23", "merge", "single_aspect", 2, "朋友合作与联机体验", ["G10"], "E1—E4 支持朋友合作，E5 含联机乐趣和延迟，E6 仅配置。与 G10 统一语义主题，延迟同时记性能。"),
    ("G24", "split", "mixed_uninterpretable", 0, "上头、时长与零散问题", ["G18"], "E1—E3 是上头或时长，不直接证明耐玩；E4 谈找枪耗时，E5 是网络及封禁，E6 是总体感受。按具体诉求拆分。"),
    ("G25", "rework", "mixed_uninterpretable", 0, "time 多义词混合", [], "E1 的 all time 是最高评价，E3 是累计时长，E2 是玩久后无聊，E4—E6 为零散体验。不能因 time 共词合成主题。"),
    ("G26", "generic", "generic_evaluation", 0, "好玩与上瘾泛赞", ["G05", "G13", "G16", "G30"], "六例都未展开具体玩法；上瘾不自动等于重玩性，统一归泛赞。"),
    ("G27", "split", "multiple_aspects", 1, "多方面长评与表达混合", ["G22"], "E1、E2 为评分长评；E3 也是有具体证据的综合长评，E4 谈画面和操作；E5、E6 语境有限。具体方面分别归类，不能将后三至四例统称泛评。"),
    ("G28", "split", "multiple_aspects", 1, "更新维护与更新后故障", ["G07"], "E1、E5 归更新后运行故障；E3、E4、E6 归内容更新或支持；E2 的慢可能指下载或发布频率，保留歧义，不强分。"),
    ("G29", "retain", "single_aspect", 2, "剧情与叙事", [], "E1—E3、E4 支持剧情，正负评价可共存；E5 为泛赞，E6 自称未充分了解剧情，不作剧情支持例。"),
    ("G30", "generic", "generic_evaluation", 0, "简短整体好评", ["G05", "G13", "G16", "G26"], "E1、E2、E4—E6 是泛赞，E3 表达版本偏好；统一泛评类别，不形成具体方面主题。"),
]


def run(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new output directory")
    parent = split.load_parent(BASE)
    sample = review.read_json(review.ROOT / "docs/review/review_sample.json")
    directory = BASE / "assignments"
    manifest = review.read_json(directory / "assignment_manifest.json")
    if manifest["parent_review_results_sha256"] != review.sha(BASE / "review_results.json"):
        raise ValueError("Parent report checksum mismatch")
    if manifest["source_fingerprint"] != parent["source_fingerprint"]:
        raise ValueError("Assignment source mismatch")
    split.verify_partition(parent, manifest["assignments"])
    paths = [directory / "steam-review-20261107-v1_adjudication_A_Cai_draft (1).json",
             directory / "steam-review-20261107-v1_adjudication_B_Lin_draft.json"]
    if tuple(review.sha(path) for path in paths) != EXPECTED_DRAFT_SHA256:
        raise ValueError("Review drafts changed; reassess the round-specific decisions before rerunning")
    watched = paths + [directory / "assignment_manifest.json", BASE / "review_results.json"]
    watched += [review.ROOT / "docs/review" / f["filename"] for f in parent["sources"]]
    hashes = {str(p): review.sha(p) for p in watched}
    decisions = review.make_template(parent)
    submissions, contributors, owner = {}, [], {}
    for path, part in zip(paths, manifest["assignments"]):
        data = review.read_json(path)
        # A draft is usable evidence but must never be relabelled a human complete file.
        if data.get("scope") != "assignment" or data.get("assignment_id") != part["id"] or data.get("assignment_sha256") != part["assignment_sha256"]:
            raise ValueError("Draft assignment mismatch")
        errors = review.decision_errors({**data, "scope": "full", "complete": False}, parent, sample)
        for stage, key in (("cases", "case_ids"), ("groups", "group_ids")):
            if set(data[stage]) != set(part[key]):
                errors.append(f"{stage}: incomplete or unexpected coverage")
            for i, row in data[stage].items():
                errors += [f"{i}: {e}" for e in review.answer_errors(row.get("final_answer"), sample, stage)]
                owner[i] = part["id"]
            decisions[stage].update(copy.deepcopy(data[stage]))
        if errors:
            raise ValueError("Invalid draft: " + "; ".join(errors))
        submissions[part["id"]] = data
        contributors.append({"assignment_id": part["id"], "file": path.relative_to(review.ROOT).as_posix(),
                             "sha256": review.sha(path), "submitted_complete": data["complete"],
                             "submitted_reviewer": data["reviewer"],
                             "submitted_acknowledged_review": data["acknowledged_review"],
                             "submitted_status_counts": {st: dict(Counter(x["status"] for x in data[st].values())) for st in ("cases", "groups")}})
    originals = {r["id"]: r for r in parent["cases"]}
    resolve_cases(decisions["cases"], originals)
    for i, action, kind, rating, name, related, note in GROUPS:
        decisions["groups"][i].update(status="reviewed", basis="custom", action=action,
            related_groups=related, final_answer={"group_kind": kind, "coherence_rating": rating,
                                                 "proposed_name": name, "notes": note}, rationale=note)
    audit = []
    for stage in ("cases", "groups"):
        for i, row in decisions[stage].items():
            submitted = submissions[owner[i]][stage][i]
            fields = [k for k in row["final_answer"] if row["final_answer"][k] != submitted["final_answer"].get(k)]
            substantive = [k for k in fields if k in review.CORE_FIELDS or k in ("group_kind", "coherence_rating")]
            audit.append({"id": i, "stage": stage, "owner": owner[i], "submitted_status": submitted["status"],
                          "submitted_basis": submitted.get("basis"), "submitted_rationale": submitted["rationale"],
                          "changed_answer_fields": fields, "substantive_changed_fields": substantive,
                          "before": {k: submitted["final_answer"].get(k) for k in fields},
                          "after": {k: row["final_answer"][k] for k in fields},
                          "submitted_action": submitted.get("action"), "final_action": row.get("action"),
                          "adjudication_rationale": row["rationale"]})
            row["review_source"] = {"assignment_id": owner[i], "submitted_status": submitted["status"]}
            row["adjudicator_type"] = "ai"
    now = datetime.now(timezone.utc).isoformat()
    decisions.update(scope="full", complete=True, reviewer="Codex（依据双方复核意见及规则裁决）",
        acknowledged_review=True, adjudicator_type="ai", human_adjudication_complete=False,
        contributors=contributors, contributors_complete=False, generated_at=now,
        task12_topics_frozen=False, ai_assistance={"role": "adjudication", "authorized_by": "用户要求按照双方复核意见进行裁决",
        "human_signoff": False, "source_draft_status_preserved": True},
        interpretation="complete 表示本轮 AI 裁决已生成，不代表两名复核者逐项点击确认或全员冻结主题。")
    errors = review.decision_errors(decisions, parent, sample)
    for i, row in decisions["cases"].items():
        for aspect, evidence in row["final_answer"]["aspect_evidence"].items():
            if evidence not in originals[i]["text"]:
                errors.append(f"{i}/{aspect}: evidence is not a continuous source quotation")
    if errors:
        raise ValueError("Invalid final decisions: " + "; ".join(errors))
    destination.mkdir(parents=True)
    final = destination / "adjudication.json"
    review.write_json(final, decisions)
    review.write_json(destination / "decision_audit.json", {"generated_at": now, "source_fingerprint": parent["source_fingerprint"],
        "input_hashes": hashes, "generator_sha256": review.sha(Path(__file__)), "records": audit})
    resolved = {"package_id": parent["package_id"], "source_fingerprint": parent["source_fingerprint"],
        "adjudication_sha256": review.sha(final), "task12_topics_frozen": False,
        "note": "180条使用裁决；其余120条仅沿用讨论前双方标签共识，并非新增人工或AI复核。",
        "cases": {i: {"answer": copy.deepcopy(decisions["cases"][i]["final_answer"] if i in decisions["cases"] else row["A"]),
                       "origin": "ai_adjudicated_after_assigned_review" if i in decisions["cases"] else "original_label_agreement",
                       "original_evidence_source": None if i in decisions["cases"] else "A"} for i, row in originals.items()},
        "groups": copy.deepcopy(decisions["groups"])}
    review.write_json(destination / "resolved_sample.json", resolved)
    payload = copy.deepcopy(parent)
    payload.update(initial_decisions=decisions, human_adjudication_complete=False, task12_topics_frozen=False,
                   status="ai_adjudication_from_review_drafts", generated_at=now,
                   parent_report_template_sha256=parent["report_template_sha256"],
                   report_template_sha256=review.sha(Path(review.__file__).with_suffix(".html")),
                   adjudicator_sha256=review.sha(Path(__file__)),
                   review_session_id=review.sha(final), decision_context="本页已载入依据双方草稿生成的 AI 裁决。原始确认状态保留在审计文件；修改不代表 Task12 主题冻结。")
    split.render(payload, destination / "裁决检查.html")
    write_summary(destination, decisions, audit, resolved)
    if any(review.sha(Path(p)) != digest for p, digest in hashes.items()):
        raise RuntimeError("An input changed during adjudication")
    return decisions, audit


def write_summary(destination, decisions, audit, resolved):
    names = {"retain": "保留候选", "merge": "统一语义主题", "split": "拆分", "rework": "重新分组", "generic": "泛评／非方面"}
    changed = [x for x in audit if x["stage"] == "cases" and x["substantive_changed_fields"]]
    groups_changed = [x for x in audit if x["stage"] == "groups" and x["substantive_changed_fields"]]
    uncertain = [i for i, row in decisions["cases"].items() if row["final_answer"]["aspect_presence"] == "uncertain"]
    lines = ["# 双方复核结果与本轮裁决", "", "已依据两份复核草稿及原文完成 180 条评论、30 个候选组的 AI 裁决；原始答卷与草稿未改写。", "",
        "[查看／修改裁决](裁决检查.html) · [裁决 JSON](adjudication.json) · [300 条解析结果](resolved_sample.json) · [逐项变更审计](decision_audit.json)", "",
        "## 提交情况", "", "| 分工 | 已填写 | 原文件已点击确认 |", "| --- | --- | --- |",
        "| A-Cai | 90 条评论、15 组 | 54 条评论、0 组 |", "| B-Lin | 90 条评论、15 组 | 0 条评论、0 组 |", "",
        "两份都是 draft，姓名与总确认均为空。已填写内容作为复核意见使用，未补签人名或把原草稿伪装成完成答卷。裁决 JSON 的 complete=true 仅表示本轮 AI 裁决完整。", "",
        "## 统一口径", "", "- 明确的朋友同玩、运行故障、催更、购买限制均可构成具体方面，不能因短句或玩梗一律归泛评。",
        "- 泛赞、怀旧和累计时长不直接构成玩法或耐玩性；候选组评分衡量具体方面，不衡量情绪或关键词相似。",
        "- 更新后启动失败归性能；误封归联机管理；没有安装 MOD 不算 MOD 评价；PLATFORMING 是平台跳跃。",
        "- 保留有依据的复核选择；改动与原复核理由均记录在审计中。无关的残留证据及已失效的原因字段已清理。", "",
        f"180 条中，{len(changed)} 条调整了语言、方面、有效原因或表达标签；其他条目沿用复核选择，部分仅修正证据或备注。{len(groups_changed)} 组调整了性质或评分。", "",
        "## 候选组裁决", "", "| 组 | 结果 | 分数 | 名称 | 依据与处理 |", "| --- | --- | ---: | --- | --- |"]
    for i, row in decisions["groups"].items():
        a = row["final_answer"]
        lines.append(f"| {i} | {names[row['action']]} | {a['coherence_rating']} | {a['proposed_name']} | {a['notes']} |")
    lines += ["", "G10/G23 统一为朋友合作与联机体验；G17/G21 只对齐难度部分。G05/G13/G16/G26/G30 统一泛赞口径，仍保留各原组编号。G18/G24 的时长、G22/G27 的长评形式不作为单一方面。", "",
        "这些是匿名候选组的语义裁决。尚无 G 编号到具体模型、K、簇的映射，因此没有修改模型簇或重跑训练。组分数仍针对当前六例，不是清理后的预估分数。", "",
        "## 人工检查重点", "", "以下是裁决中的边界判断，已给出结论或保留不确定，不阻塞文件使用：", "",
        "- R063：眩晕归身体舒适度；R108：重复战斗同时保留玩法与内容；R130/R189：笼统玩法评分不扩展为操作或具体机制。",
        "- R137：战斗评级与 emoji；R187：招架诉求；R202：时长不直接算耐玩性；R214/R215：专名或字母标识不自动算中英混合。",
        "- G06、G08、G27：原复核把含多个具体方面的例子概括成泛评，已纠正；G11、G13、G18、G24 的方面评分已统一。",
        "- G28-E2：“更新慢”未区分下载速度和发布频率，保留歧义。", "",
        "裁决后方面仍不确定的记录：" + "、".join(uncertain) + "。", "",
        "## 评论标签变更", "", "| 编号 | 原复核人 | 调整字段 | 裁决理由 |", "| --- | --- | --- | --- |"]
    for x in changed:
        lines.append(f"| {x['id']} | {x['owner']} | {', '.join(x['substantive_changed_fields'])} | {x['adjudication_rationale']} |")
    counts = Counter(x["answer"]["aspect_presence"] for x in resolved["cases"].values())
    lines += ["", "## 后续状态", "", f"供后续使用的 300 条结果：有方面 {counts['yes']}、无方面 {counts['no']}、不确定 {counts['uncertain']}。其中 120 条沿用原始双方标签共识，未另作裁决。",
        "讨论前的一致率、κ 和原始统计保持不变；不把裁决结果当成独立标注信度。Task12 尚未冻结，仍需模型映射、稳定性检查及全员确认。", ""]
    (destination / "裁决说明.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions, audit = run(args.output)
    print({"cases": len(decisions["cases"]), "groups": len(decisions["groups"]),
           "case_label_changes": sum(x["stage"] == "cases" and bool(x["substantive_changed_fields"]) for x in audit)})
