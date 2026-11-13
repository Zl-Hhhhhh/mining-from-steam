"""Recover deterministic tree input order from the fixed sampler, without refitting.

The first run recorded sample membership but not linkage row order separately.
Independent validation checks the recovered order against all six stored tree cuts.
"""
from pathlib import Path
import csv
import json
import cluster_bge as pipeline


def main():
    config=json.loads((Path(__file__).parent/"config.json").read_text(encoding="utf-8"))
    root=pipeline.ROOT;out=root/"output/clustering-bge/freeze_20261101-v1"
    p=root/"data/topic-formal/freeze_20261101/8381107/model_reviews.jsonl"
    rows=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    rows=[r for r in rows if r["text_clean"].strip()]
    fit=pipeline.fit_indices(rows)
    sample=pipeline.balanced_sample(fit,rows,config["common_sample_size"],config["sampling_seed"])
    order=[str(rows[i]["recommendationid"]) for i in sample]
    with (out/"scope_index.csv").open(encoding="utf-8-sig",newline="") as f:
        recorded={r["recommendationid"] for r in csv.DictReader(f) if r["in_common5000"]=="True"}
    assert len(order)==5000 and set(order)==recorded
    pipeline.dump(out/"common_review_order.json",order)
    pipeline.dump(out/"scope_order_provenance.json",{"method":"Reconstructed after initial run using unchanged deterministic sampler, source ID order, seed and config; no model refit. Independent validation compares tree cuts with stored membership labels.","refit":False,"pipeline_sha256":pipeline.sha_file(Path(pipeline.__file__)),"scope_size":5000})
    print("Recorded 5000 linkage input IDs; no clustering repeated.")


if __name__=="__main__": main()
