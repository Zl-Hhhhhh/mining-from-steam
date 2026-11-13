"""Prove a complete cache is usable while model-encoder imports are prohibited."""
import builtins
import json
from pathlib import Path
import embedding_bge as bge

HERE = Path(__file__).resolve().parent
ROOT = bge.ROOT
MAIN = ROOT / "output/embedding-bge/freeze_20261101-v1"
OUT = HERE / "cache-reuse-check"
original = builtins.__import__


def guarded(name, *args, **kwargs):
    if name == "sentence_transformers" or name.startswith("sentence_transformers."):
        raise AssertionError("Cache reuse attempted to import the encoder")
    return original(name, *args, **kwargs)


if __name__ == "__main__":
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    builtins.__import__ = guarded
    try:
        bge.run(config, OUT)
    finally:
        builtins.__import__ = original
    s = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))
    main = json.loads((MAIN / "run_summary.json").read_text(encoding="utf-8"))
    assert s["cache_was_complete_on_entry"] is True
    assert s["vectors_sha256"] == main["vectors_sha256"]
    assert (OUT / "review_index.csv").read_bytes() == (MAIN / "review_index.csv").read_bytes()
    result = {"status":"passed", "encoder_import_prohibited":True, "cache_reused":True,
              "identical_review_index":True,"identical_vectors_sha256":True}
    (MAIN / "cache_reuse_validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))
