"""Step 3: offline BGE-M3 whole-review embeddings, resumable and ID-aligned."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "config/topic_formal.json").exists()),
            Path(os.environ.get("STEAM_PROJECT_ROOT", r"D:\我的\课程\Intro. to Data Mining\project\mining-from-steam")))
DEFAULT_CONFIG = ROOT / "config/embedding_bge.json"
if not DEFAULT_CONFIG.exists():
    DEFAULT_CONFIG = HERE / "config.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def encoder_texts(rows):
    return [r["text_clean"] for r in rows]


def unique_text_plan(rows):
    unique, lookup, review_to_unique = [], {}, []
    for row in rows:
        text = row["text_clean"]
        if not text.strip():
            raise ValueError("Empty text has no embedding")
        if text not in lookup:
            lookup[text] = len(unique)
            unique.append(text)
        review_to_unique.append(lookup[text])
    return unique, review_to_unique


def batch_indices(indices, lengths, max_batch, token_budget):
    """Length-sorted batches; an input longer than budget is allowed only alone."""
    batch = []
    for i in sorted(indices, key=lambda i: (lengths[i], i)):
        if batch and (len(batch) == max_batch or lengths[i] * (len(batch) + 1) > token_budget):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def validate_vectors(matrix, expected_rows, dimension):
    import numpy as np
    if matrix.shape != (expected_rows, dimension) or matrix.dtype != np.float32:
        raise ValueError("Wrong vector shape or dtype")
    for start in range(0, expected_rows, 2048):
        b = matrix[start:start + 2048]
        if not np.isfinite(b).all() or not np.allclose(np.linalg.norm(b, axis=1), 1, atol=2e-5):
            raise ValueError("Invalid vectors")


def peak_memory_gib():
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ["peak", "working", "quota_peak_paged", "quota_paged", "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile"]]
    m = PMC(); m.cb = ctypes.sizeof(m)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    p = ctypes.WinDLL("psapi", use_last_error=True)
    p.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(m), m.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return m.peak / 2**30


def load_inputs(config):
    base = ROOT / "data/topic-formal" / config["freeze_id"] / config["source_commit"][:7]
    manifest = json.loads((base / "prepared_manifest.json").read_text(encoding="utf-8"))
    if manifest["source_commit"] != config["source_commit"]:
        raise ValueError("Wrong frozen source")
    for entry in manifest["local_files"]:
        if sha_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Prepared input changed: " + entry["path"])
    source = base / "model_reviews.jsonl"
    if sha_file(source) != config["input_sha256"]:
        raise ValueError("Normalized input hash changed")
    rows = read_jsonl(source)
    master = read_jsonl(base / "snapshot/data/cleaned" / config["freeze_id"] / "clean_master.jsonl")
    if len(rows) != 19395 or len(master) != 19486:
        raise ValueError("Frozen denominator changed")
    ids = [str(r["recommendationid"]) for r in rows]
    master_ids = [str(r["recommendationid"]) for r in master]
    if len(set(ids)) != len(ids) or len(set(master_ids)) != len(master_ids) or not set(ids) <= set(master_ids):
        raise ValueError("Unsafe or inconsistent IDs")
    encoded = [r for r in rows if r["text_clean"].strip()]
    return base, manifest, rows, master, encoded


def run(config, output):
    import numpy as np
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer
    versions = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "sentence-transformers", "numpy", "huggingface-hub"]}
    base, prepared, rows, master, encoded = load_inputs(config)
    unique, review_to_unique = unique_text_plan(encoded)
    snapshot = ROOT / ".cache-topic/models" / ("models--" + config["model_name"].replace("/", "--")) / "snapshots" / config["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    token_ids = tokenizer(unique, add_special_tokens=True, truncation=False,
                          return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    lengths = [len(x) for x in token_ids]
    if max(lengths) > config["max_seq_length"]:
        raise ValueError("Input exceeds selected model limit; no silent truncation")
    review_lengths = [lengths[i] for i in review_to_unique]
    identity = {"model": config["model_name"], "revision": config["model_revision"],
                "review_order_and_text": digest([[r["recommendationid"], r["text_clean"]] for r in encoded]),
                "source_input_sha256": config["input_sha256"], "max_seq_length": config["max_seq_length"],
                "prompt": config["prompt"], "pooling": config["pooling"], "dtype": config["dtype"],
                "normalize": config["normalize_embeddings"], "representation": config["representation"],
                "versions": versions}
    cache_key = digest(identity)
    cache = ROOT / ".cache-topic/embeddings-bge-formal" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "run_summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old["cache_key"] != cache_key:
            raise ValueError("Output belongs to a different configuration")
    meta_path = cache / "manifest.json"
    vector_path = cache / "review_vectors.npy"
    uid_path = cache / "unique_vectors.npy"
    done_path = cache / "completed.npy"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["identity"] != identity or meta["unique_text_sha256"] != [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in unique]:
            raise ValueError("Cache identity mismatch")
    else:
        meta = {"status":"in_progress", "created_at":now(), "identity":identity,
                "unique_text_sha256":[hashlib.sha256(t.encode("utf-8")).hexdigest() for t in unique],
                "pilot_reused_unique":0, "pilot_reused_review_ids":[], "encoding_seconds":0.0,
                "review_ids":[str(r["recommendationid"]) for r in encoded], "cache_key":cache_key}
        u = np.lib.format.open_memmap(uid_path, mode="w+", dtype=np.float32, shape=(len(unique), config["embedding_dimension"]))
        u[:] = np.nan; u.flush(); del u
        d = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(len(unique),))
        d[:] = False; d.flush(); del d
        dump(meta_path, meta)
    cache_was_complete = meta["status"] == "complete"
    uv = np.load(uid_path, mmap_mode="r+", allow_pickle=False)
    done = np.load(done_path, mmap_mode="r+", allow_pickle=False)
    if uv.shape != (len(unique), 1024) or done.shape != (len(unique),):
        raise ValueError("Partial cache shape mismatch")
    if done.any():
        validate_vectors(np.asarray(uv[done]), int(done.sum()), config["embedding_dimension"])
    if cache_was_complete:
        if sha_file(vector_path) != meta["vectors_sha256"]:
            raise ValueError("Completed cache checksum mismatch")
        validate_vectors(np.load(vector_path, mmap_mode="r", allow_pickle=False), len(encoded), 1024)
        print("Complete ID-aligned cache verified; no model inference", flush=True)
    else:
        # Reuse pilot vectors only when every semantic setting matches and texts fit
        # both sequence limits. Raising the limit changes none of their tokens.
        pilot = ROOT / config["pilot_directory"]
        pm = json.loads((pilot / "bge_m3/summary.json").read_text(encoding="utf-8"))
        for field, expected in [("model", config["model_name"]), ("revision", config["model_revision"]), ("prompt", ""), ("dtype", "float32"), ("normalize", True), ("versions", versions)]:
            if pm[field] != expected:
                raise ValueError("Pilot is incompatible: " + field)
        pv_path = Path(pm["cache_path"]) / "embeddings.npy"
        pcm = json.loads((pv_path.parent / "manifest.json").read_text(encoding="utf-8"))
        pr = json.loads((pilot / "inputs.json").read_text(encoding="utf-8"))
        if sha_file(pv_path) != pm["vectors_sha256"] or pcm["ids"] != [r["id"] for r in pr]:
            raise ValueError("Pilot vectors not aligned")
        pv = np.load(pv_path, allow_pickle=False)
        validate_vectors(pv, len(pr), 1024)
        lookup = {t:i for i,t in enumerate(unique)}
        source_by_id = {str(r["recommendationid"]):r for r in encoded}
        reused_ids = set(meta["pilot_reused_review_ids"])
        for i, r in enumerate(pr):
            if r["kind"] != "real":
                continue
            rid = r["recommendationid"]
            if rid not in source_by_id or source_by_id[rid]["text_clean"] != r["text"]:
                raise ValueError("Pilot/source text mismatch")
            j = lookup[r["text"]]
            if lengths[j] > pm["max_tokens"]:
                raise ValueError("Pilot would have truncated this source")
            if not done[j]:
                uv[j] = pv[i]; done[j] = True
                meta["pilot_reused_unique"] += 1
            reused_ids.add(rid)
        meta["pilot_reused_review_ids"] = sorted(reused_ids)
        uv.flush(); done.flush(); dump(meta_path, meta)
        pending = np.flatnonzero(~done).tolist()
        batches = list(batch_indices(pending, lengths, config["max_batch_size"], config["max_padded_tokens_per_batch"]))
        dump(output / "encoding_plan.json", {"cache_key":cache_key, "source_rows":len(master),
             "nonempty_source_rows":len(rows), "encoded_reviews":len(encoded), "unique_texts":len(unique),
             "unique_texts_pending":len(pending), "pilot_reused_unique_texts":meta["pilot_reused_unique"],
             "pilot_reused_review_ids":len(reused_ids), "batches":len(batches),
             "review_tokens":{"p50":float(np.percentile(review_lengths,50)), "p95":float(np.percentile(review_lengths,95)),
                              "p99":float(np.percentile(review_lengths,99)), "max":max(review_lengths),
                              "over_512":sum(n>512 for n in review_lengths)}, "truncated":0})
        print(json.dumps({"unique_texts":len(unique), "pilot_reused":meta["pilot_reused_unique"], "remaining":len(pending), "batches":len(batches), "max_tokens":max(lengths)}), flush=True)
        if pending:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(config["cpu_threads"]); torch.set_num_interop_threads(1)
            model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True, trust_remote_code=False,
                                        model_kwargs={"dtype":torch.float32,"attn_implementation":"sdpa","weights_only":True})
            model.max_seq_length = config["max_seq_length"]
            if config["prompt"] != "":
                raise ValueError("This BGE representation requires the explicit empty prompt used in the pilot")
            # Recent SentenceTransformers versions may expose optional named
            # prompts even for this model. Explicit prompt="" overrides them,
            # just as in the selected pilot; optional prompts are not features.
            started = time.perf_counter(); logged = started; base_seconds = meta["encoding_seconds"]
            for number, batch in enumerate(batches, 1):
                v = model.encode([unique[i] for i in batch], prompt="", batch_size=len(batch),
                                 show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
                validate_vectors(v, len(batch), 1024)
                uv[batch] = v; uv.flush()
                done[batch] = True; done.flush()
                elapsed = time.perf_counter() - started
                if time.perf_counter() - logged >= 20 or number == len(batches):
                    meta.update(encoding_seconds=base_seconds+elapsed, completed_unique_texts=int(done.sum()), last_checkpoint_at=now())
                    dump(meta_path, meta)
                    print(json.dumps({"completed_unique":int(done.sum()), "total_unique":len(unique),
                                      "batch":number,"batches":len(batches),"last_max_tokens":max(lengths[i] for i in batch),
                                      "encoding_seconds":round(meta["encoding_seconds"],1)}), flush=True)
                    logged = time.perf_counter()
            del model
        if not done.all():
            raise ValueError("Incomplete embeddings")
        validate_vectors(uv, len(unique), 1024)
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32, shape=(len(encoded),1024))
        for start in range(0, len(encoded), 1024):
            vectors[start:start+1024] = uv[review_to_unique[start:start+1024]]
        vectors.flush(); del vectors
        meta.update(status="complete", completed_at=now(), vectors_sha256=sha_file(vector_path),
                    unique_vectors_sha256=sha_file(uid_path), completed_unique_texts=int(done.sum()))
        dump(meta_path, meta)
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    validate_vectors(vectors, len(encoded), 1024)
    encoded_lookup = {str(r["recommendationid"]):i for i,r in enumerate(encoded)}
    normalized = {str(r["recommendationid"]):r for r in rows}
    index = []
    for r in master:
        rid = str(r["recommendationid"]); nr = normalized.get(rid)
        vi = encoded_lookup.get(rid)
        status = "encoded" if vi is not None else ("empty_after_markup_cleanup" if nr is not None else "source_empty")
        index.append({"recommendationid":rid, "appid":r["appid"],
                      "source_language":r["language"], "vector_index":vi if vi is not None else "",
                      "status":status,"text_sha256":hashlib.sha256(nr["text_clean"].encode("utf-8")).hexdigest() if nr else "",
                      "token_length":review_lengths[vi] if vi is not None else "",
                      "source_fit_eligible":nr["fit_eligible"] if nr else False,
                      "source_low_information":nr["source_low_information"] if nr else "",
                      "emoji_candidate":nr["emoji_candidate"] if nr else False,
                      "nonlexical":nr["nonlexical"] if nr else True})
    with (output / "review_index.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
    dump(cache / "review_ids.json",[str(r["recommendationid"]) for r in encoded])
    dump(output / "config.json",config)
    summary = {"status":"step3_complete_clustering_pending", "completed_at":now(), "model_name":config["model_name"],
               "model_revision":config["model_revision"],"config":config,"versions":versions,
               "source_commit":config["source_commit"],"source_manifest_sha256":sha_file(base / "prepared_manifest.json"),
               "source_input_sha256":config["input_sha256"],"cache_key":cache_key,"cache_path":str(cache),
               "vector_file":str(vector_path),"vector_shape":list(vectors.shape),"vectors_sha256":sha_file(vector_path),
               "source_master_reviews":len(master),"source_nonempty_reviews":len(rows),"encoded_reviews":len(encoded),
               "unique_texts":len(unique),"status_counts":dict(Counter(r["status"] for r in index)),
               "pilot_reused_unique_texts":meta["pilot_reused_unique"],"pilot_reused_review_ids":len(meta["pilot_reused_review_ids"]),
               "encoding_seconds":meta["encoding_seconds"],"cache_was_complete_on_entry":cache_was_complete,
               "peak_process_working_set_gib":peak_memory_gib(),"max_observed_tokens":max(review_lengths),
               "over_512_reviews":sum(n>512 for n in review_lengths),"truncated_reviews":0,
               "index_sha256":sha_file(output / "review_index.csv"),"pipeline_sha256":sha_file(Path(__file__)),
               "clustering_performed":False,"human_theme_validation_complete":False,
               "limitations":["Original API language labels remain unaudited; encoding does not correct them.",
                              "A whole-review vector can mix multiple aspects. Long-context support is not topic validation.",
                              "No vocabulary/style/emoji filter removes otherwise nonempty source text.",
                              "Empty comments retain IDs but have no zero/fake vector. Source counts remain traceable.",
                              "CSV restoration issue remains separate; the verified JSONL input is used.",
                              "No clustering, final theme codebook or RQ conclusions generated at Step 3."]}
    dump(summary_path,summary)
    print(json.dumps({"complete":True,"encoded_reviews":len(encoded),"shape":list(vectors.shape),
                      "encoding_seconds":meta["encoding_seconds"],"output":str(output)},ensure_ascii=True),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--output",type=Path,default=ROOT / "output/embedding-bge/freeze_20261101-v1")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")),args.output)


if __name__ == "__main__":
    main()
