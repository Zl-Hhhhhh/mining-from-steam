"""Apply generic rejection to the relaxed Task12 topic matching trial."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

import embedding_bge_sentences as encoder


ROOT = encoder.ROOT
WORK = ROOT / ".cache-topic/lexical-relaxed-20261125"
AUDIT = ROOT / "output/topic_review/task12-relaxed-blind-20261125"
DEFAULT_POLICY = ROOT / "config/topic_generic_rejection.json"
DEFAULT_EMBEDDINGS = WORK / "encoding_config.json"
DEFAULT_BANK = WORK / "generic_bank.npz"
DEFAULT_SCORES = WORK / "topic-matching-probe/new_unit_matches.jsonl"
MODEL_FIELDS = ("model_name", "model_revision", "model_files_sha256", "pooling",
                "normalize_embeddings", "dtype", "max_seq_length")
VARIANTS = ("before_rejection", "rule_only", "rule_and_semantic")
MARKS = re.compile(r"[.!?。！？；;,，…]+")
TONE = r"(?:really |very |so |absolutely |genuinely |pretty |extremely |fking |fucking )?"
ADJECTIVE = r"(?:good|great|nice|fun|awesome|amazing|excellent|fantastic|impressive|wonderful|bad|terrible|awful)"
ENGLISH = re.compile(
    rf"(?:it(?:'?s| is| was)? |this(?: game)? (?:is|was) )?(?:such )?(?:an? )?{TONE}"
    rf"{ADJECTIVE}(?: and {ADJECTIVE})?(?: game)?"
    r"|(?:an? |the )?(?:(?:absolute|classic|genuine|true) )?masterpiece"
    r"|i (?:love|loved|like|liked|hate|hated|enjoy|enjoyed) (?:it|this game|the game)(?: so much| a lot)?"
    r"|(?:i )?(?:would )?(?:highly |definitely )?recommend (?:it|this game)"
    r"|one of (?:the |my )?(?:best|worst|favourite|favorite) games(?: of all time)?"
)
CHINESE = re.compile(
    r"(?:这个游戏|这游戏)?(?:真的|太|真|特别|十分|超级|非常|实在|太太)?(?:好玩|好棒|棒|烂|差)(?:了|啊)?"
    r"|(?:真的|真的是|绝对是)?(?:个)?神作|(?:我)?(?:很|非常|特别|真)?喜欢(?:这款游戏|这个游戏|这游戏)"
    r"|垃圾游戏|值得一玩|值得玩|强烈推荐|推荐入手|别玩了|不推荐"
)


def is_pure_generic(text: str) -> bool:
    from prepare_task12_handoff import GENERIC_REVIEW

    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().replace("’", "'").split())

    def generic_phrase(phrase):
        return any(pattern.fullmatch(phrase) for pattern in (GENERIC_REVIEW, ENGLISH, CHINESE))

    if generic_phrase(normalized.strip(" .!?。！？…")):
        return True
    clauses = [chunk.strip() for chunk in MARKS.split(normalized) if chunk.strip()]
    return bool(clauses) and all(generic_phrase(clause) for clause in clauses)


def routes(text: str, topic: str | None, topic_score: float, generic_score: float, policy: dict):
    if is_pure_generic(text):
        return ("A01", "A01")
    if topic is None:
        return ("A03", "A03")
    if topic in policy["semantic_topics"] and topic_score - generic_score < policy["minimum_topic_over_generic_margin"]:
        return (None, "A03")
    return (None, None)


def rows(path):
    with path.open(encoding="utf-8") as handle:
        yield from map(json.loads, handle)


def bank_metadata(policy, config):
    examples = [{"group": group, "text": text}
                for group, texts in policy["generic_examples"].items() for text in texts]
    if not examples or len({row["text"] for row in examples}) != len(examples):
        raise ValueError("Generic examples must be nonempty and unique")
    return {"examples": examples, "model": {field: config[field] for field in MODEL_FIELDS}}


def encode_bank(policy, config, output):
    if output.exists():
        raise FileExistsError(f"Generic bank already exists: {output}")
    metadata = bank_metadata(policy, config)
    texts = [row["text"] for row in metadata["examples"]]
    model = encoder.load_encoder(config)
    vectors = np.asarray(model.encode(
        texts, prompt="", batch_size=config["max_batch_size"],
        show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True,
    ), dtype=np.float32)
    encoder.validate_vectors(vectors, len(texts))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, vectors=vectors, metadata=json.dumps(metadata, ensure_ascii=False))
    temporary.replace(output)
    print(f"Encoded {len(texts)} generic examples into {output}")


def compare_audit(predictions, key):
    answer_path = AUDIT / "reviewer_A/task12-relaxed-blind-20261125_A_complete.json"
    answer = encoder.read_json(answer_path)
    if (answer["complete"] is not True or answer["assignment"] != "A"
            or answer["package_id"] != key["package_id"] or answer["sample_sha256"] != key["sample_sha256"]
            or answer["answers"].keys() != key["cases"].keys()):
        raise ValueError("Reviewer A's answer file does not match the blind sample")
    counters = {variant: Counter() for variant in VARIANTS}
    for case_id, case in key["cases"].items():
        judgement = answer["answers"][case_id]
        if judgement["judgment"] == "A03":
            continue
        gold = {topic for topic in judgement["evidence"] if topic.startswith("T")}
        baseline = set(case["baseline_topic_ids"])
        rid = case["review_id"]
        old_added = predictions["before_rejection"].get(rid, set()) - baseline
        for variant, counter in counters.items():
            added = predictions[variant].get(rid, set()) - baseline
            kept = baseline | added
            counter["eligible_reviews"] += 1
            counter["true_additions"] += len(added & gold)
            counter["unsupported_additions"] += len(added - gold)
            counter["generic_reviews_with_new_topics"] += judgement["judgment"] == "A01" and bool(added)
            counter["supported_additions_lost_vs_before"] += len((old_added - added) & gold)
            counter["unsupported_additions_removed_vs_before"] += len((old_added - added) - gold)
            counter["true_pairs_total"] += len(kept & gold)
            counter["unsupported_pairs_total"] += len(kept - gold)
            counter["missing_pairs_total"] += len(gold - kept)
    return {variant: dict(counter) for variant, counter in counters.items()}


def apply(policy, embeddings_path, bank_path, scores_path, output):
    if output.exists():
        raise FileExistsError(f"Choose a new result directory: {output}")
    encoder.verify(embeddings_path)
    config, _, units, _ = encoder.load_inputs(embeddings_path)
    vector_dir = ROOT / config["output"]
    with np.load(bank_path, allow_pickle=False) as bank:
        negatives = bank["vectors"]
        metadata = json.loads(bank["metadata"].item())
    if metadata != bank_metadata(policy, config):
        raise ValueError("Generic bank examples or model settings differ")
    encoder.validate_vectors(negatives, len(metadata["examples"]))

    scores = list(rows(scores_path))
    if len(scores) != len(units) or any(
        (score["unit_id"], score["text"], score["review_ids"]) != (unit["unit_id"], unit["text"], unit["review_ids"])
        for score, unit in zip(scores, units)
    ):
        raise ValueError("Topic rows and the verified sentence vectors are not aligned")
    if any(unit["fit_candidate"] for unit in units):
        raise ValueError("This experiment expects only newly eligible application units")
    audited_key = encoder.read_json(AUDIT / "coordinator/sampling_key.json")
    recorded_sha = audited_key["inputs"][DEFAULT_SCORES.relative_to(ROOT).as_posix()]
    scores_sha = encoder.sha_file(scores_path)
    if scores_sha != recorded_sha:
        raise ValueError("Scores differ from the trial used to draw the A sample")
    predictions = {variant: defaultdict(set) for variant in VARIANTS}
    counts = {variant: Counter() for variant in VARIANTS}
    vectors = np.load(vector_dir / "sentence_vectors.npy", mmap_mode="r", allow_pickle=False)
    closest_indices = []
    closest_scores = []
    for start in range(0, len(vectors), 2048):
        distances = vectors[start:start + 2048] @ negatives.T
        closest_indices.extend(np.argmax(distances, axis=1).tolist())
        closest_scores.extend(np.max(distances, axis=1).tolist())
    output.mkdir(parents=True)
    with (output / "unit_decisions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for scored, generic_score, closest in zip(scores, closest_scores, closest_indices):
            topic = scored["global_top_topic"] if policy["base_policy"] in scored["passed_policies"] else None
            score = scored["global_top_score"]
            rule_route, semantic_route = routes(scored["text"], topic, score, generic_score, policy)
            labels = {
                "before_rejection": topic,
                "rule_only": topic if rule_route is None else None,
                "rule_and_semantic": topic if semantic_route is None else None,
            }
            for variant, label in labels.items():
                if label:
                    counts[variant][label] += 1
                    for rid in scored["review_ids"]:
                        predictions[variant][rid].add(label)
            decision = {
                "unit_id": scored["unit_id"], "text": scored["text"], "review_ids": scored["review_ids"],
                "topics_by_variant": labels, "rule_route": rule_route, "semantic_route": semantic_route,
                "top_topic": scored["global_top_topic"], "top_topic_score": score,
                "generic_score": round(generic_score, 6), "topic_over_generic": round(score - generic_score, 6),
                "closest_generic_group": metadata["examples"][closest]["group"],
            }
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    summary = {
        "status": "development_comparison_with_reviewer_A",
        "units": len(units), "generic_bank_examples": len(negatives),
        "thresholds": {key: policy[key] for key in ("base_policy", "semantic_topics", "minimum_topic_over_generic_margin")},
        "score_file_sha256": scores_sha, "generic_bank_sha256": encoder.sha_file(bank_path),
        "variants": {variant: {"accepted_units": sum(counts[variant].values()), "topic_counts": counts[variant],
                               "reviews_with_new_assignments": len(predictions[variant])} for variant in VARIANTS},
        "reviewer_A": compare_audit(predictions, audited_key),
    }
    encoder.write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["encode-bank", "apply"])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--embeddings-config", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=WORK / "generic-rejection-probe")
    args = parser.parse_args()
    policy = encoder.read_json(args.policy)
    if args.action == "encode-bank":
        encode_bank(policy, encoder.read_json(args.embeddings_config), args.bank)
    else:
        apply(policy, args.embeddings_config, args.bank, args.scores, args.output)


if __name__ == "__main__":
    main()
