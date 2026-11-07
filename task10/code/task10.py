"""Portable Task 10 entry: verify, download fixed weights, encode, cluster.

No pilot cache, old checkout, absolute machine path or model weights are required
to verify/use the delivered vectors. Encoding is a separate explicit action.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def js(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");t.replace(p)
def load_inputs():
    c=js(ROOT/"config/embedding.json")
    for name,expected in c["input_checksums"].items():
        if sha(ROOT/name)!=expected:raise ValueError("Input changed: "+name)
    rows=[json.loads(x) for x in (ROOT/"inputs/model_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    return c,[r for r in rows if r["text_clean"].strip()]
def verify():
    import numpy as np
    c,rows=load_inputs();m=js(ROOT/"artifacts/embedding_manifest.json")
    if any(m[k]!=c[k] for k in ['model_name','model_revision']) or m['source_input_sha256']!=c['input_checksums']['inputs/model_reviews.jsonl']:raise ValueError('Model/input manifest mismatch')
    x=np.load(ROOT/m["vector_file"],mmap_mode="r",allow_pickle=False)
    ids=js(ROOT/"artifacts/review_ids.json")
    if ids!=[str(r["recommendationid"]) for r in rows] or len(set(ids))!=len(ids) or x.shape!=(len(rows),1024) or x.dtype!=np.float32:raise ValueError("Vector-ID alignment or dtype error")
    if sha(ROOT/m["vector_file"])!=m["vectors_sha256"]:raise ValueError("Vector checksum mismatch")
    for i in range(0,len(x),2048):
        a=x[i:i+2048]
        if not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=2e-5):raise ValueError("Invalid vectors")
    result={"status":"passed","reviews":len(rows),"shape":list(x.shape),"model":m["model_name"],"model_revision":m["model_revision"],"encoder_loaded":False,"paths":"bundle-relative"}
    print(json.dumps(result));return result

def download():
    from huggingface_hub import snapshot_download
    c,_=load_inputs();path=snapshot_download(c["model_name"],revision=c["model_revision"],token=False,cache_dir=ROOT/".model-cache",allow_patterns=["config.json","config_sentence_transformers.json","modules.json","1_Pooling/config.json","sentence_bert_config.json","pytorch_model.bin","sentencepiece.bpe.model","special_tokens_map.json","tokenizer.json","tokenizer_config.json"])
    print("Downloaded fixed public BGE weights to local cache: "+str(path))

def encode(offline=False):
    import numpy as np
    c,rows=load_inputs()
    if (ROOT/"artifacts/review_vectors.npy").exists():
        return verify()  # Does not import the encoder or require old checkpoints.
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    if offline:os.environ["HF_HUB_OFFLINE"]="1"
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(c["cpu_threads"])
    model_path=snapshot_download(c["model_name"],revision=c["model_revision"],token=False,cache_dir=ROOT/".model-cache",local_files_only=offline,allow_patterns=["*.json","1_Pooling/config.json","pytorch_model.bin","sentencepiece.bpe.model"])
    model=SentenceTransformer(model_path,device="cpu",local_files_only=True,trust_remote_code=False,model_kwargs={"dtype":torch.float32,"attn_implementation":"sdpa","weights_only":True})
    model.max_seq_length=8192
    unique=list(dict.fromkeys(r["text_clean"] for r in rows));lookup={t:i for i,t in enumerate(unique)}
    lengths=[len(model.tokenizer(t,truncation=False)["input_ids"]) for t in unique]
    if max(lengths)>8192:raise ValueError("No silent truncation: input exceeds native limit")
    fingerprint=hashlib.sha256(json.dumps({"model":c["model_revision"],"texts":unique,"dtype":"float32","limit":8192},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    checkpoint=ROOT/".encoding-work"/fingerprint;checkpoint.mkdir(parents=True,exist_ok=True)
    vp=checkpoint/"unique.npy";dp=checkpoint/"done.npy"
    if not dp.exists():
        x=np.lib.format.open_memmap(vp,mode="w+",dtype=np.float32,shape=(len(unique),1024));x[:]=np.nan;x.flush();del x
        d=np.lib.format.open_memmap(dp,mode="w+",dtype=bool,shape=(len(unique),));d[:]=False;d.flush();del d
    x=np.load(vp,mmap_mode="r+",allow_pickle=False);done=np.load(dp,mmap_mode="r+",allow_pickle=False)
    pending=sorted(np.flatnonzero(~done),key=lambda i:(lengths[i],int(i)));batches=[];batch=[]
    for i in pending:
        if batch and (len(batch)==8 or lengths[i]*(len(batch)+1)>4096):batches.append(batch);batch=[]
        batch.append(int(i))
    if batch:batches.append(batch)
    started=time.perf_counter()
    for n,b in enumerate(batches,1):
        v=model.encode([unique[i] for i in b],prompt="",batch_size=len(b),show_progress_bar=False,normalize_embeddings=True,convert_to_numpy=True)
        if not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=2e-5):raise ValueError("Invalid batch")
        x[b]=v;x.flush();done[b]=True;done.flush()
        if n%20==0 or n==len(batches):print(json.dumps({"batch":n,"batches":len(batches),"seconds":round(time.perf_counter()-started,1)}),flush=True)
    if not done.all():raise ValueError("Incomplete encoding")
    artifacts=ROOT/"artifacts";artifacts.mkdir(exist_ok=True)
    assembling=artifacts/"review_vectors.assembling.npy"
    out=np.lib.format.open_memmap(assembling,mode="w+",dtype=np.float32,shape=(len(rows),1024))
    for start in range(0,len(rows),1024):out[start:start+1024]=x[[lookup[r["text_clean"]] for r in rows[start:start+1024]]]
    out.flush();del out
    assembling.replace(artifacts/"review_vectors.npy")
    dump(artifacts/"review_ids.json",[str(r["recommendationid"]) for r in rows])
    actual=sha(artifacts/"review_vectors.npy")
    dump(artifacts/"embedding_manifest.json",{"model_name":c["model_name"],"model_revision":c["model_revision"],"vector_file":"artifacts/review_vectors.npy","cache_path":"artifacts","cache_key":fingerprint,"vectors_sha256":actual,"shape":[len(rows),1024],"source_input_sha256":c["input_checksums"]["inputs/model_reviews.jsonl"],"encoding_seconds_this_session":time.perf_counter()-started,"new_reproduction":True})
    cc=js(ROOT/"config/clustering.json");cc["vectors_sha256"]=actual;dump(ROOT/"config/clustering-reproduced.json",cc)
    print("Encoding complete. For this reproduced cache, use clustering-reproduced.json; numerical hashes can differ across environments.")
    verify()

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["verify","download","encode","cluster"]);p.add_argument("--offline",action="store_true");p.add_argument("--output",type=Path);p.add_argument("--cluster-config",type=Path)
    a=p.parse_args()
    if a.action=="verify":verify()
    elif a.action=="download":download()
    elif a.action=="encode":encode(a.offline)
    else:
        verify();command=[sys.executable,"-B",str(ROOT/"code/cluster_bge_portable.py"),"--output",str((a.output or ROOT/"results/reproduced").resolve())]
        if a.cluster_config:command += ["--config",str(a.cluster_config.resolve())]
        subprocess.run(command,check=True,cwd=ROOT)

if __name__=="__main__":main()
