"""Improve readable examples without altering fitted models or memberships."""
from pathlib import Path
import csv
import json
import numpy as np
import cluster_bge as p

HERE=Path(__file__).resolve().parent
ROOT=p.ROOT
OUT=ROOT/"output/clustering-bge/freeze_20261101-v1"

def csvs(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def main():
    s=json.loads((OUT/"run_summary.json").read_text(encoding="utf-8"));config=s["config"]
    metrics=[json.loads((folder/"metrics.json").read_text(encoding="utf-8")) for folder in OUT.iterdir() if folder.is_dir() and (folder/"metrics.json").exists()]
    # Preserve the original presentation order.
    order=[r["candidate"] for r in csvs(OUT/"candidate_metrics.csv")];metrics.sort(key=lambda r:order.index(r["candidate"]))
    rows=[json.loads(x) for x in (ROOT/"data/topic-formal/freeze_20261101/8381107/model_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    rows=[r for r in rows if r["text_clean"].strip()];byid={str(r["recommendationid"]):i for i,r in enumerate(rows)}
    profiles=[];reps=[];compact={};hashes={};examples_checked=0
    for m in metrics:
        key=m["candidate"];folder=OUT/key
        hashes[key]=p.sha_file(folder/"assignments.csv")
        a=csvs(folder/"assignments.csv")
        labels=[-1]*len(rows)
        for r in a:
            if r["recommendationid"] in byid:labels[byid[r["recommendationid"]]]=int(r["raw_cluster"])
        compact[key]=labels
        old=csvs(folder/"representative_reviews.csv")
        chosen=[]
        for c in range(m["k"]):
            members=[r for r in a if r["used_for_fit"]=="True" and int(r["raw_cluster"])==c]
            members.sort(key=lambda r:(-float(r["cosine_to_own_centroid"]),r["recommendationid"]))
            for selection,lang in [("center",None),("center_english","english"),("center_schinese","schinese")]:
                seen=set();n=0;target=config["representatives_center"] if lang is None else config["representatives_per_language"]
                for ar in members:
                    r=rows[byid[ar["recommendationid"]]]
                    if lang and r["language"]!=lang:continue
                    if r["text_clean"] in seen:continue
                    seen.add(r["text_clean"])
                    chosen.append({"candidate":key,"cluster":c,"selection":selection,"recommendationid":str(r["recommendationid"]),"language":r["language"],"game_name":r["game_name"],"text":r["text_clean"],"cosine_to_own_centroid":float(ar["cosine_to_own_centroid"])})
                    n+=1
                    if n==target:break
            # Keep the originally fixed random IDs; no cherry-picking examples.
            chosen.extend({**r,"cluster":int(r["cluster"]),"cosine_to_own_centroid":float(r["cosine_to_own_centroid"])} for r in old if int(r["cluster"])==c and r["selection"]=="random")
        for r in chosen:
            assert labels[byid[r["recommendationid"]]]==r["cluster"]
            examples_checked+=1
        p.csv_out(folder/"representative_reviews.csv",chosen);reps.extend(chosen)
        pp=csvs(folder/"cluster_profiles.csv")
        for r in pp:
            profiles.append({k:(int(v) if k in ["cluster","fit_count","applied_count","fit_english","fit_schinese"] else float(v) if k.endswith("_share") else v) for k,v in r.items()})
    p.csv_out(OUT/"representative_reviews.csv",reps)
    p.render_report(rows,metrics,profiles,reps,compact,s,OUT)
    # Explain the presentation change adjacent to the representative panel.
    report=OUT/"report.html";page=report.read_text(encoding="utf-8")
    page=page.replace('<h2>2. 选择候选，阅读中心与随机评论</h2>','<h2>2. 选择候选，阅读中心与随机评论</h2><p class="small">中心例子按完全相同文本去重，方便阅读；随机例子保持原抽样，拟合和簇规模没有去重。所有主题名仍为空，重复评论各自的ID继续保留。</p>')
    report.write_text(page,encoding="utf-8")
    assert all(p.sha_file(OUT/k/"assignments.csv")==h for k,h in hashes.items())
    p.dump(OUT/"representative_validation.json",{"status":"passed","examples_checked":examples_checked,"memberships_changed":False,"center_exact_text_dedup_only":True,"original_random_examples_preserved":True,"refit":False})
    print(json.dumps({"representative_examples_checked":examples_checked,"refit":False}))

if __name__=="__main__":main()
