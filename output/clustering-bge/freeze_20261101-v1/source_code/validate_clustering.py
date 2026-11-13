"""Independent checks of scopes, published diagnostics and untouched cache."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score,silhouette_score
from threadpoolctl import threadpool_limits

ROOT=Path(r"D:\我的\课程\Intro. to Data Mining\project\mining-from-steam")
OUT=ROOT/"output/clustering-bge/freeze_20261101-v1"

def js(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def csvs(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    checks=[]
    def check(name,condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    s=js(OUT/"run_summary.json");es=js(ROOT/"output/embedding-bge/freeze_20261101-v1/run_summary.json")
    vectors=np.load(es["vector_file"],mmap_mode="r",allow_pickle=False)
    check("Step 3 vectors unchanged; no encoder or new embeddings",sha(es["vector_file"])==es["vectors_sha256"]==s["embedding_vectors_sha256"] and s["encoder_loaded"] is False and s["new_embeddings_computed"]==0)
    scope=csvs(OUT/"scope_index.csv");byid={r["recommendationid"]:r for r in scope}
    ids=js(Path(es["cache_path"])/"review_ids.json")
    ids_set=set(ids)
    check("Scope index follows cached IDs exactly",[r["recommendationid"] for r in scope]==ids)
    full=[r["recommendationid"] for r in scope if r["used_for_full_fit"]=="True"]
    common_set={r["recommendationid"] for r in scope if r["in_common5000"]=="True"}
    eval_ids=[r["recommendationid"] for r in scope if r["in_silhouette1500"]=="True"]
    check("13145 lexical fit / 5000 common / 1500 evaluation nested",len(full)==13145 and len(common_set)==5000 and len(eval_ids)==1500 and set(eval_ids)<=common_set<=set(full))
    check("Common sample has 2500 per API language",Counter(byid[r]["source_language"] for r in common_set)=={"english":2500,"schinese":2500})
    base=ROOT/"data/topic-formal/freeze_20261101/8381107"
    source=[json.loads(x) for x in (base/"model_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    selected={str(r["recommendationid"]) for r in source if r["fit_eligible"] and r["text_clean"].strip() and not r["nonlexical"]}
    check("Fit matches source eligibility, excluding symbols and cleanup-empty",set(full)==selected)
    master=[json.loads(x) for x in (base/"snapshot/data/cleaned/freeze_20261101/clean_master.jsonl").read_text(encoding="utf-8").splitlines()]
    master_ids=[str(r["recommendationid"]) for r in master];votes={str(r["recommendationid"]):r["voted_up"] for r in master}
    metrics=csvs(OUT/"candidate_metrics.csv")
    check("15 candidates: six full KMeans, nine common-scope models",len(metrics)==15 and Counter(m["scope"] for m in metrics)=={"full_fit":6,"common5000":9})
    evaluation_indices=[int(byid[r]["vector_index"]) for r in eval_ids]
    distances=squareform(pdist(vectors[evaluation_indices],metric="cosine"))
    # Linkage rows are in the deterministic sampled order, not vector order.
    # Obtain that order from one hierarchy's representative-independent stored
    # scope artifact written separately alongside the linkage.
    common_order=js(OUT/"common_review_order.json")
    check("Tree order is the exact common-scope ID set",len(common_order)==5000 and set(common_order)==common_set)
    cross_checks=[]
    for m in metrics:
        key=m["candidate"];k=int(m["k"]);a=csvs(OUT/key/"assignments.csv");am={r["recommendationid"]:r for r in a}
        check(key+": every master ID retained in order",[r["recommendationid"] for r in a]==master_ids)
        used={r["recommendationid"] for r in a if r["used_for_fit"]=="True"}
        expected=set(full) if m["scope"]=="full_fit" else common_set
        applied_expected=ids_set if m["scope"]=="full_fit" else common_set
        check(key+": exact fit and application scope",used==expected and all((int(r["raw_cluster"])>=0)==(r["recommendationid"] in applied_expected) for r in a))
        labels=[int(am[r]["raw_cluster"]) for r in expected]
        counts=Counter(labels)
        check(key+": k nonempty clusters, correct sizes",len(counts)==k and min(counts.values())==int(m["smallest_cluster"]) and max(counts.values())==int(m["largest_cluster"]))
        profiles=csvs(OUT/key/"cluster_profiles.csv")
        valid=True
        for p in profiles:
            members=[rid for rid in expected if int(am[rid]["raw_cluster"])==int(p["cluster"])]
            valid &= len(members)==int(p["fit_count"])
            valid &= abs(sum(votes[rid] for rid in members)/len(members)-float(p["fit_recommended_share"]))<1e-10
            valid &= p["human_theme_name"]==""
        check(key+": profile sizes and post-fit recommendation shares",bool(valid))
        labels_eval=[int(am[r]["raw_cluster"]) for r in eval_ids]
        actual=float(silhouette_score(distances,labels_eval,metric="precomputed")) if 1<len(set(labels_eval))<len(labels_eval) else None
        reported=float(m["silhouette_cosine_common1500"]) if m["silhouette_cosine_common1500"] else None
        check(key+": independently recomputed shared-sample silhouette",actual is None and reported is None or actual is not None and reported is not None and abs(actual-reported)<5e-6)
        if m["algorithm"] in ["ward_euclidean","average_cosine"]:
            prefix="ward" if m["algorithm"].startswith("ward") else "average"
            tree=np.load(OUT/(prefix+"_linkage.npy"),allow_pickle=False)
            recut=fcluster(tree,k,criterion="maxclust")
            stored=[int(am[r]["raw_cluster"]) for r in common_order]
            check(key+": stored memberships match independent tree cut",adjusted_rand_score(recut,stored)>1-1e-12)
        cross_checks.append({"candidate":key,"silhouette_difference":abs(actual-reported) if actual is not None else None})
    check("No fitted recommendation target or final theme claim",s["recommendation_labels_used_for_fitting_or_selection"] is False and "step5_human_review_pending" in s["status"])
    result={"status":"passed","check_count":len(checks),"checks":checks,"metric_recomputation":cross_checks,"independent_human_semantic_validation":False}
    (OUT/"validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","check_count":len(checks)},ensure_ascii=True))

if __name__=="__main__":
    with threadpool_limits(limits=4): main()
