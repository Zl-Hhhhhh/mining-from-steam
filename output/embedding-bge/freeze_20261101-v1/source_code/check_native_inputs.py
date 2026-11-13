"""Verify actual encoder tokenization and short-text pilot compatibility."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
ROOT = Path(r"D:\我的\课程\Intro. to Data Mining\project\mining-from-steam")
OUT = ROOT / "output/embedding-bge/freeze_20261101-v1"


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    summary=json.loads((OUT/"run_summary.json").read_text(encoding="utf-8"))
    snapshot=ROOT/".cache-topic/models/models--BAAI--bge-m3/snapshots"/summary["model_revision"]
    model=SentenceTransformer(str(snapshot),device="cpu",local_files_only=True,trust_remote_code=False,
                              model_kwargs={"dtype":torch.float32,"attn_implementation":"sdpa","weights_only":True})
    model.max_seq_length=8192
    source=[json.loads(x) for x in (ROOT/"data/topic-formal/freeze_20261101/8381107/model_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    source=[r for r in source if r["text_clean"].strip()]
    lengths=[len(model.tokenizer(r["text_clean"],truncation=False)["input_ids"]) for r in source]
    chosen=sorted(range(len(source)),key=lambda i:lengths[i])[-3:]
    input_batch=model.tokenize([source[i]["text_clean"] for i in chosen])
    actual=input_batch["attention_mask"].sum(dim=1).tolist()
    expected=[lengths[i] for i in chosen]
    assert actual==expected and max(actual)==3993
    for pos,i in enumerate(chosen):
        raw=model.tokenizer(source[i]["text_clean"],truncation=False)["input_ids"]
        kept=input_batch["input_ids"][pos][input_batch["attention_mask"][pos].bool()].tolist()
        assert raw==kept
    pilot=ROOT/"experiments/embedding-model-pilot-20261105/results-v1"
    pr=json.loads((pilot/"inputs.json").read_text(encoding="utf-8"))
    ps=json.loads((pilot/"bge_m3/summary.json").read_text(encoding="utf-8"))
    old=np.load(Path(ps["cache_path"])/"embeddings.npy",allow_pickle=False)
    picked=[next(i for i,r in enumerate(pr) if r["kind"]=="real" and r["language"]==lang) for lang in ["english","schinese"]]
    fresh=model.encode([pr[i]["text"] for i in picked],prompt="",batch_size=2,
                       normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
    max_delta=float(np.max(np.abs(fresh-old[picked])))
    assert np.allclose(fresh,old[picked],atol=1e-5,rtol=0)
    result={"status":"passed","actual_long_encoder_input_tokens":actual,
            "raw_and_actual_encoder_token_ids_equal":True,"native_limit":8192,
            "pilot_short_text_vectors_recomputed":2,"short_text_max_absolute_delta":max_delta,
            "tolerance":1e-5,"full_dataset_reencoded":False}
    (OUT/"native_input_validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
