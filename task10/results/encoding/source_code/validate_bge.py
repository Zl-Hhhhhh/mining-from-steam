"""Independent Step 3 cache/index verification without loading model weights."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

ROOT = Path(r"D:\我的\课程\Intro. to Data Mining\project\mining-from-steam")
OUT = ROOT / "output/embedding-bge/freeze_20261101-v1"
BASE = ROOT / "data/topic-formal/freeze_20261101/8381107"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    s = js(OUT / "run_summary.json")
    cache = Path(s["cache_path"])
    meta = js(cache / "manifest.json")
    prepared = js(BASE / "prepared_manifest.json")
    sources = [json.loads(x) for x in (BASE / "model_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    master = [json.loads(x) for x in (BASE / "snapshot/data/cleaned/freeze_20261101/clean_master.jsonl").read_text(encoding="utf-8").splitlines()]
    encoded = [r for r in sources if r["text_clean"].strip()]
    with (OUT / "review_index.csv").open(encoding="utf-8-sig",newline="") as f:
        index = list(csv.DictReader(f))
    matrix = np.load(cache / "review_vectors.npy",mmap_mode="r",allow_pickle=False)
    unique = np.load(cache / "unique_vectors.npy",mmap_mode="r",allow_pickle=False)
    done = np.load(cache / "completed.npy",allow_pickle=False)
    ids = js(cache / "review_ids.json")
    check("Selected fixed BGE revision and completed Step 3 status", s["model_name"]=="BAAI/bge-m3" and s["model_revision"]=="5617a9f61b028005a4858fdac845db406aefb181" and meta["status"]=="complete")
    downloaded=js(ROOT/"experiments/embedding-model-pilot-20261105/models_downloaded.json")["bge_m3"]
    weight=ROOT/".cache-topic/models/models--BAAI--bge-m3/snapshots"/s["model_revision"]/"pytorch_model.bin"
    check("Local model weights match the fixed pilot download fingerprint", sha(weight)==downloaded["files"]["pytorch_model.bin"]["sha256"])
    check("Prepared input and frozen source manifest hashes match", sha(BASE/"model_reviews.jsonl")==s["source_input_sha256"] and sha(BASE/"prepared_manifest.json")==s["source_manifest_sha256"])
    check("All independently prepared input files remain unchanged", all(sha(ROOT/e["path"])==e["sha256"] for e in prepared["local_files"]))
    check("19486 master IDs retained in original order", [r["recommendationid"] for r in index]==[str(r["recommendationid"]) for r in master] and len(index)==19486)
    check("Review-index checksum matches execution record", sha(OUT/"review_index.csv")==s["index_sha256"])
    check("Explicit 19392 encoded / 91 source-empty / 3 cleanup-empty status", dict(Counter(r["status"] for r in index))=={"encoded":19392,"source_empty":91,"empty_after_markup_cleanup":3})
    check("Encoded vector IDs match source ordering and cache manifest", ids==[str(r["recommendationid"]) for r in encoded]==meta["review_ids"])
    check("19392 by 1024 float32 vectors, no zero placeholders", matrix.shape==(19392,1024) and matrix.dtype==np.float32)
    check("Every vector is finite and unit-normalized", bool(np.isfinite(matrix).all() and np.max(np.abs(np.linalg.norm(matrix,axis=1)-1))<2e-5))
    check("Review vectors checksum matches both records", sha(cache/"review_vectors.npy")==s["vectors_sha256"]==meta["vectors_sha256"])
    check("Unique-text vectors checksum and completion markers verified", sha(cache/"unique_vectors.npy")==meta["unique_vectors_sha256"] and bool(done.all()) and len(done)==s["unique_texts"])
    lookup = {str(r["recommendationid"]):r for r in encoded}
    cache_positions = {rid:i for i,rid in enumerate(ids)}
    mapped = True
    for r in index:
        rid = r["recommendationid"]
        if r["status"]=="encoded":
            mapped &= int(r["vector_index"])==cache_positions[rid]
            mapped &= r["text_sha256"]==hashlib.sha256(lookup[rid]["text_clean"].encode("utf-8")).hexdigest()
        else:
            mapped &= r["vector_index"]=="" and r["token_length"]==""
    check("Every master-to-vector mapping and text hash verified", mapped)
    unique_lookup = {h:i for i,h in enumerate(meta["unique_text_sha256"])}
    identical = all(np.array_equal(matrix[i],unique[unique_lookup[hashlib.sha256(r["text_clean"].encode("utf-8")).hexdigest()]]) for i,r in enumerate(encoded))
    check("Exact-text reuse preserves every individual review vector", identical)
    pilot = ROOT / "experiments/embedding-model-pilot-20261105/results-v1"
    psummary = js(pilot / "bge_m3/summary.json")
    pv = np.load(Path(psummary["cache_path"])/"embeddings.npy",allow_pickle=False)
    prows = js(pilot/"inputs.json")
    real = [(i,r) for i,r in enumerate(prows) if r["kind"]=="real"]
    check("All 600 reused real pilot vectors match exactly", len(real)==600 and all(np.array_equal(pv[i],matrix[cache_positions[r["recommendationid"]]]) for i,r in real))
    check("Index tokens within native limit, all 139 >512 reviews retained", max(int(r["token_length"]) for r in index if r["status"]=="encoded")==3993 and sum(int(r["token_length"])>512 for r in index if r["status"]=="encoded")==139 and s["truncated_reviews"]==0)
    check("Representation is full-review, not hidden chunk averaging", s["config"]["max_seq_length"]==8192 and "untruncated whole-review" in s["config"]["representation"])
    check("No clustering or completed theme validation claimed", s["clustering_performed"] is False and s["human_theme_validation_complete"] is False)
    result={"status":"passed","check_count":len(checks),"checks":checks,"semantic_theme_accuracy_validated":False}
    (OUT/"validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=True))


if __name__=="__main__":
    main()
