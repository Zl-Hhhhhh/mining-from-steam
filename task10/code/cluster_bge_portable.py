"""Step 4: cluster verified BGE vectors; never load an embedding encoder."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = ROOT / "config/clustering.json"


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+".tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");t.replace(path)


def csv_out(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def script_hint(text):
    han=len(re.findall(r"[\u3400-\u9fff]",text));latin=len(re.findall(r"[A-Za-z]",text))
    letters=sum(ch.isalpha() for ch in text)
    if han>=2 and han/max(1,letters)>=.4: return "Han-dominant"
    if han==0 and latin>=4 and latin/max(1,letters)>=.8: return "Latin-script"
    return "other_or_uncertain"


def fit_indices(rows):
    import numpy as np
    return np.array([i for i,r in enumerate(rows) if r["fit_eligible"] and r["text_clean"].strip() and not r["nonlexical"]],dtype=int)


def balanced_sample(indices,rows,size,seed):
    """Equal source-language quotas, game round robin; source labels stay unchanged."""
    import numpy as np
    rng=np.random.default_rng(seed);selected=[]
    languages=sorted({rows[i]["language"] for i in indices})
    for pos,lang in enumerate(languages):
        groups=defaultdict(list)
        for i in indices:
            if rows[i]["language"]==lang: groups[rows[i]["appid"]].append(int(i))
        for group in groups.values(): rng.shuffle(group)
        target=size//len(languages)+(pos<size%len(languages))
        language_selected=[]
        while len(language_selected)<target:
            added=False
            for key in sorted(groups):
                if groups[key] and len(language_selected)<target:
                    language_selected.append(groups[key].pop());added=True
            if not added: raise ValueError("Not enough data for shared balanced sample")
        selected.extend(language_selected)
    return np.array(selected,dtype=int)


def validate_alignment(rows,ids,vectors):
    import numpy as np
    if [str(r["recommendationid"]) for r in rows]!=ids or len(set(ids))!=len(ids):
        raise ValueError("Embedding ID order or uniqueness mismatch")
    if vectors.shape!=(len(rows),1024) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-5):
        raise ValueError("Embedding matrix shape/finiteness/norm mismatch")


def centers_and_scores(X,labels,k):
    import numpy as np
    centers=np.stack([X[labels==c].mean(axis=0) for c in range(k)]).astype(np.float32)
    unit=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-12)
    sims=X@unit.T
    own=sims[np.arange(len(X)),labels]
    masked=sims.copy();masked[np.arange(len(X)),labels]=-np.inf
    gap=own-masked.max(axis=1)
    return centers,own,gap


def metric_row(identifier,algorithm,scope,k,seed,indices,labels,rows,votes,eval_indices,eval_dist,config):
    import numpy as np
    from sklearn.metrics import adjusted_mutual_info_score,silhouette_score
    counts=Counter(int(x) for x in labels)
    lookup={int(i):int(c) for i,c in zip(indices,labels)}
    sampled=np.array([lookup[int(i)] for i in eval_indices])
    silhouette=float(silhouette_score(eval_dist,sampled,metric="precomputed")) if 1<len(set(sampled))<len(sampled) else None
    ys=[rows[i]["language"] for i in indices]
    scripts=[script_hint(rows[i]["text_clean"]) for i in indices]
    tags=[votes[str(rows[i]["recommendationid"])] for i in indices]
    return {"candidate":identifier,"algorithm":algorithm,"scope":scope,"k":k,"seed":seed,
            "fit_reviews":len(indices),"silhouette_cosine_common1500":silhouette,
            "smallest_cluster":min(counts.values()),"largest_cluster":max(counts.values()),
            "largest_cluster_share":max(counts.values())/len(indices),
            "clusters_under20":sum(n<config["small_cluster_threshold"] for n in counts.values()),
            "reviews_in_clusters_under20":sum(n for n in counts.values() if n<config["small_cluster_threshold"]),
            "source_language_ami":float(adjusted_mutual_info_score(ys,labels)),
            "writing_system_ami":float(adjusted_mutual_info_score(scripts,labels)),
            "game_ami":float(adjusted_mutual_info_score([rows[i]["appid"] for i in indices],labels)),
            "steam_recommendation_ami_diagnostic":float(adjusted_mutual_info_score(tags,labels)),
            "theme_status":"unvalidated_candidate_clusters"}


def export_candidate(identifier,algorithm,scope,k,seed,fit_idx,fit_labels,applied_idx,applied_labels,
                     vectors,rows,master_ids,votes,eval_idx,eval_dist,output,config,raw_centers=None):
    import numpy as np
    folder=output/identifier;folder.mkdir()
    centers,_,_=centers_and_scores(vectors[fit_idx],fit_labels,k)
    if raw_centers is not None: centers=raw_centers.astype(np.float32)
    unit=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-12)
    sims=vectors[applied_idx]@unit.T
    own=sims[np.arange(len(applied_idx)),applied_labels]
    masked=sims.copy();masked[np.arange(len(applied_idx)),applied_labels]=-np.inf
    gaps=own-masked.max(axis=1)
    np.savez_compressed(folder/"centers.npz",raw_centers=centers,unit_centers=unit)
    fit_set=set(map(int,fit_idx));label_by_index={int(i):int(c) for i,c in zip(applied_idx,applied_labels)}
    detail_by_index={int(i):(float(s),float(g)) for i,s,g in zip(applied_idx,own,gaps)}
    by_id={str(r["recommendationid"]):i for i,r in enumerate(rows)}
    assignments=[]
    for rid in master_ids:
        i=by_id.get(rid);label=label_by_index.get(i,-1)
        if i is None: status="empty_text"
        elif label<0: status="outside_common_sample"
        elif rows[i]["nonlexical"]: status="nonlexical_no_concrete_theme"
        else: status="unvalidated_cluster_candidate"
        score,gap=detail_by_index.get(i,(None,None))
        candidate=label if status=="unvalidated_cluster_candidate" else -1
        assignments.append({"recommendationid":rid,"raw_cluster":label,"candidate_cluster":candidate,
                            "status":status,"used_for_fit":i in fit_set,
                            "cosine_to_own_centroid":score,"centroid_cosine_gap":gap,
                            "boundary_reading_flag":bool(gap is not None and gap<config["boundary_cosine_gap"])})
    csv_out(folder/"assignments.csv",assignments)
    profile=[];reps=[];rng=np.random.default_rng(config["sampling_seed"])
    for c in range(k):
        fit_members=[int(i) for i,lab in zip(fit_idx,fit_labels) if lab==c]
        applied=[int(i) for i,lab in zip(applied_idx,applied_labels) if lab==c]
        lc=Counter(rows[i]["language"] for i in fit_members)
        gc=Counter(str(rows[i]["appid"]) for i in fit_members)
        profile.append({"candidate":identifier,"cluster":c,"fit_count":len(fit_members),
                        "applied_count":len(applied),"fit_english":lc.get("english",0),"fit_schinese":lc.get("schinese",0),
                        "fit_english_share":lc.get("english",0)/len(fit_members),
                        "fit_recommended_share":sum(votes[str(rows[i]["recommendationid"])] for i in fit_members)/len(fit_members),
                        "top_game_share":max(gc.values())/len(fit_members),
                        "applied_low_information_share":sum(rows[i]["source_low_information"] for i in applied)/len(applied),
                        "applied_short_le10_share":sum(len(rows[i]["text_clean"])<=10 for i in applied)/len(applied),
                        "applied_emoji_share":sum(rows[i]["emoji_candidate"] for i in applied)/len(applied),
                        "human_theme_name":""})
        central=sorted(fit_members,key=lambda i:(-detail_by_index[i][0],rows[i]["recommendationid"]))
        selected=[("center",i) for i in central[:config["representatives_center"]]]
        selected += [("random",int(i)) for i in rng.choice(fit_members,min(config["representatives_random"],len(fit_members)),replace=False)]
        for lang in ["english","schinese"]:
            selected += [("center_"+lang,i) for i in central if rows[i]["language"]==lang][:config["representatives_per_language"]]
        for selection,i in selected:
            reps.append({"candidate":identifier,"cluster":c,"selection":selection,
                         "recommendationid":str(rows[i]["recommendationid"]),"language":rows[i]["language"],
                         "game_name":rows[i]["game_name"],"text":rows[i]["text_clean"],
                         "cosine_to_own_centroid":detail_by_index[i][0]})
    csv_out(folder/"cluster_profiles.csv",profile);csv_out(folder/"representative_reviews.csv",reps)
    metric=metric_row(identifier,algorithm,scope,k,seed,fit_idx,fit_labels,rows,votes,eval_idx,eval_dist,config)
    dump(folder/"metrics.json",metric)
    print(json.dumps(metric,ensure_ascii=True),flush=True)
    return metric,profile,reps,applied_labels


def make_plots(metrics,profiles,output,config):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=[("common_kmeans_k10_s17","KMeans"),("ward_k10","Ward / Euclidean"),("average_k10","Average / cosine")]
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    for ax,(key,title) in zip(axs.flat[:3],methods):
        ps=[p for p in profiles if p["candidate"]==key]
        cs=np.arange(len(ps));en=[p["fit_english"] for p in ps];zh=[p["fit_schinese"] for p in ps]
        ax.bar(cs,en,label="API English",color="#4774ba");ax.bar(cs,zh,bottom=en,label="API schinese",color="#de9e49")
        for i,p in enumerate(ps): ax.text(i,p["fit_count"],str(p["fit_count"]),ha="center",va="bottom",fontsize=8)
        ax.set(title=title+" (same 5,000 reviews, K=10)",xlabel="Candidate cluster ID",ylabel="Reviews")
        ax.set_xticks(cs);ax.legend(fontsize=8)
    for alg in ["kmeans","ward_euclidean","average_cosine"]:
        ms=[m for m in metrics if m["scope"]=="common5000" and m["algorithm"]==alg]
        axs[1,1].plot([m["k"] for m in ms],[m["silhouette_cosine_common1500"] for m in ms],marker="o",label=alg)
    axs[1,1].set(title="Cosine silhouette: same 1,500 evaluation reviews",xlabel="K",ylabel="Silhouette (not topic accuracy)");axs[1,1].legend(fontsize=8)
    fig.savefig(output/"algorithm_comparison.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    ps=[p for p in profiles if p["candidate"]=="full_kmeans_k10_s17"]
    cs=np.arange(10)
    axs[0].bar(cs,[p["fit_english"] for p in ps],label="API English",color="#4774ba")
    axs[0].bar(cs,[p["fit_schinese"] for p in ps],bottom=[p["fit_english"] for p in ps],label="API schinese",color="#de9e49")
    axs[0].set(title="Full-fit KMeans anchor: language counts",xlabel="Candidate cluster ID",ylabel="Fit reviews");axs[0].legend()
    axs[1].bar(cs,[p["fit_recommended_share"] for p in ps],color="#739e91")
    axs[1].set(title="Steam recommendation share (not sentiment gold)",xlabel="Candidate cluster ID",ylabel="Recommended fraction",ylim=(0,1))
    fig.savefig(output/"full_kmeans_diagnostics.png",dpi=150);plt.close(fig)


def render_report(rows,metrics,profiles,reps,compact_labels,summary,output):
    packed_rows=[{"id":str(r["recommendationid"]),"language":r["language"],"game":r["game_name"],
                  "text":r["text_clean"],"fit":bool(r["fit_eligible"] and not r["nonlexical"]),
                  "lowinfo":r["source_low_information"],"emoji":r["emoji_candidate"],"nonlexical":r["nonlexical"]} for r in rows]
    payload={"rows":packed_rows,"metrics":metrics,"profiles":profiles,"reps":reps,"labels":compact_labels,"summary":summary}
    data=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BGE Step 4 聚类比较</title><style>
body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f3f6f8;color:#172638;margin:0}main{max-width:1320px;margin:auto;padding:26px}h1{font-size:29px}h2{font-size:22px;margin-top:30px}.note{background:#fff3d0;border-left:4px solid #d5a22c;padding:15px}.card{background:white;border:1px solid #d8e0e6;border-radius:9px;padding:18px;margin:14px 0}img{max-width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border-bottom:1px solid #dfe5ea;vertical-align:top;text-align:left}th{background:#e7eef3;white-space:nowrap}.small{font-size:13px;color:#516479}.text{white-space:pre-wrap;overflow-wrap:anywhere}select,input,button{font:inherit;padding:7px;margin:5px;max-width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}summary{cursor:pointer;font-weight:600}td.comment{width:65%;min-width:260px}.rep{border-bottom:1px solid #dfe5ea;margin:9px 0;padding:8px}a{color:#285b93}@media(max-width:800px){main{padding:14px}.grid{grid-template-columns:1fr}table{font-size:14px}}</style>
<main><h1>BGE Step 4：KMeans 与层次聚类</h1><p>直接复用Step 3向量。本轮无重新编码，无正式主题命名；簇编号只表示候选分组。</p><div class="note">公平对照使用同一批5,000条评论，所有轮廓系数在其中同一1,500条上计算。全拟合集KMeans另用13,145条拟合，并应用到19,392条有向量的评论。层次聚类只报告共同样本的直接成员关系，不用中心近邻冒充新评论的层次预测。原API语言标签仍待审阅；Steam推荐标签只作拟合后的诊断，不等于情绪金标准。</div>
<h2>1. 同范围算法比较</h2><p>Ward使用单位向量上的欧氏距离；平均连接使用余弦距离。两棵树分别切成6、10、14组。预先用K=10、seed=17作浏览入口，未自动宣布赢家。</p><img src="algorithm_comparison.png" alt="共同5000条评论的算法比较"><details><summary>查看全部15个候选的指标</summary><div class="scroll" id="metrics"></div></details>
<h2>2. 选择候选，阅读中心与随机评论</h2><label>候选 <select id="candidate"></select></label><div id="scope" class="card"></div><div class="scroll" id="profiles"></div><label>候选簇 <select id="cluster"><option value="">全部</option></select></label><label>来源语言 <select id="language"><option value="">全部</option><option value="english">english</option><option value="schinese">schinese</option></select></label><label>搜索 <input id="search" placeholder="例如：优化、story、匹配"></label><p><label><input type="checkbox" id="onlyfit">仅看用于该候选拟合的评论</label><label><input type="checkbox" id="lowinfo">仅看源低信息标记</label><label><input type="checkbox" id="emoji">仅看emoji候选</label></p><div id="reps" class="card"></div><p id="count"></p><div class="scroll"><table><thead><tr><th>簇 / 范围</th><th>评论</th><th>来源信息</th></tr></thead><tbody id="reviews"></tbody></table></div><button id="more">再看50条</button>
<h2>3. 全拟合集KMeans的语言与推荐标签诊断</h2><img src="full_kmeans_diagnostics.png" alt="KMeans语言与推荐标签诊断"><p class="small">高关联只提示可能受到语言或评价方向影响，不能直接认定它是错误或正确主题。写作系统诊断区分汉字为主、拉丁字母为主及其他/不明，不等于已识别真实语言。纯符号/emoji仍保留原始分配作诊断，但不自动解释为具体主题；空文本另在主表索引中保留。</p>
<h2>4. 进入主题核验之前</h2><p>先看每组中心和随机例子是否反复谈相同具体关注点，还是泛赞、泛骂、语言或游戏名。长评可能同时谈多个方面；本轮整评向量和单一簇编号不能证明它只有一个主题。</p><p>本轮不把相似度、中心差距或轮廓系数当作置信概率/主题准确率；不依据推荐标签调K或选择模型。下一步再由成员核验候选主题、合并/拆分与no_clear_theme范围。完整数据、范围索引、代表评论和检查记录在本目录。</p></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=x=>document.getElementById(x),E=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),pct=x=>(x*100).toFixed(1)+'%';
const names={kmeans:'KMeans',ward_euclidean:'Ward层次 / 欧氏',average_cosine:'平均连接层次 / 余弦'};
for(const m of D.metrics){let o=document.createElement('option');o.value=m.candidate;o.textContent=(m.scope==='common5000'?'共同5000':'全拟合集')+' · '+names[m.algorithm]+' · K='+m.k+(m.seed!==null?' · seed='+m.seed:'');$('candidate').appendChild(o)}$('candidate').value='common_kmeans_k10_s17';
$('metrics').innerHTML='<table><tr><th>候选</th><th>拟合数</th><th>轮廓</th><th>最小簇</th><th>最大簇占比</th><th>小于20条的簇数</th><th>来源语言AMI</th><th>推荐标签AMI</th></tr>'+D.metrics.map(m=>`<tr><td>${E(m.candidate)}</td><td>${m.fit_reviews}</td><td>${m.silhouette_cosine_common1500?.toFixed(3)??'不可算'}</td><td>${m.smallest_cluster}</td><td>${pct(m.largest_cluster_share)}</td><td>${m.clusters_under20}</td><td>${m.source_language_ami.toFixed(3)}</td><td>${m.steam_recommendation_ami_diagnostic.toFixed(3)}</td></tr>`).join('')+'</table>';
let limit=50;
function candidateChanged(){let key=$('candidate').value,m=D.metrics.find(x=>x.candidate===key);$('cluster').innerHTML='<option value="">全部</option>'+Array.from({length:m.k},(_,i)=>`<option value="${i}">C${i}</option>`).join('');$('onlyfit').checked=false;limit=50;$('scope').textContent=m.scope==='common5000'?'此候选的拟合与直接分组范围：共同5,000条；其他评论不在这次算法比较内。':'此候选使用13,145条拟合，应用到19,392条有向量评论；94条空文本保留在主表索引中。';let ps=D.profiles.filter(x=>x.candidate===key);$('profiles').innerHTML='<table><tr><th>簇</th><th>拟合数</th><th>应用数</th><th>英文来源占比</th><th>推荐占比（诊断）</th><th>应用中低信息占比</th></tr>'+ps.map(p=>`<tr><td>C${p.cluster}</td><td>${p.fit_count}</td><td>${p.applied_count}</td><td>${pct(p.fit_english_share)}</td><td>${pct(p.fit_recommended_share)}</td><td>${pct(p.applied_low_information_share)}</td></tr>`).join('')+'</table>';render()}
function render(){let key=$('candidate').value,c=$('cluster').value,lang=$('language').value,q=$('search').value.toLowerCase(),m=D.metrics.find(x=>x.candidate===key),labs=D.labels[key];let selected=[];D.rows.forEach((r,i)=>{let label=labs[i];if(label<0)return;if(c!==''&&label!==Number(c))return;if(lang&&r.language!==lang)return;if(q&&!r.text.toLowerCase().includes(q))return;if($('onlyfit').checked&&!r.fit)return;if($('lowinfo').checked&&!r.lowinfo)return;if($('emoji').checked&&!r.emoji)return;selected.push([r,label])});$('count').textContent=`符合筛选 ${selected.length} 条，当前显示 ${Math.min(limit,selected.length)} 条。`; $('reviews').innerHTML=selected.slice(0,limit).map(([r,l])=>`<tr><td>C${l}<div class="small">${r.fit?'源拟合资格':'应用评论'}${r.nonlexical?' · 非词汇，不作主题':''}</div></td><td class="comment text">${E(r.text)}</td><td>${E(r.game)}<div class="small">${E(r.language)} · ${E(r.id)}</div>${r.lowinfo?'<div class="small">源低信息标记</div>':''}</td></tr>`).join('');let rr=D.reps.filter(r=>r.candidate===key&&(c===''?r.cluster===0:r.cluster===Number(c))&&(!lang||r.language===lang)&&['center','random'].includes(r.selection));$('reps').innerHTML='<strong>当前簇的中心与随机例子'+(c===''?'（默认C0）':'')+'</strong>'+rr.map(r=>`<div class="rep"><div class="small">${r.selection==='center'?'中心':'随机'} · C${r.cluster} · ${E(r.language)} · ${E(r.game_name)}</div><div class="text">${E(r.text)}</div></div>`).join('')}
$('candidate').addEventListener('change',candidateChanged);for(const id of ['cluster','language','search','onlyfit','lowinfo','emoji'])$(id).addEventListener('input',()=>{limit=50;render()});$('more').onclick=()=>{limit+=50;render()};candidateChanged();
</script></html>'''.replace("__DATA__",data)
    (output/"report.html").write_text(page,encoding="utf-8")


def run(config,output):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.pairwise import cosine_distances
    from scipy.cluster.hierarchy import linkage,cut_tree
    from scipy.spatial.distance import pdist
    from threadpoolctl import threadpool_limits
    if output.exists(): raise FileExistsError("Output exists; use a new version rather than overwrite")
    es=json.loads((ROOT/config["embedding_summary"]).read_text(encoding="utf-8"))
    es["vector_file"]=str(ROOT/es["vector_file"]) if not Path(es["vector_file"]).is_absolute() else es["vector_file"]
    es["cache_path"]=str(ROOT/es["cache_path"]) if not Path(es["cache_path"]).is_absolute() else es["cache_path"]
    if es["model_revision"]!=config["model_revision"] or sha_file(es["vector_file"])!=config["vectors_sha256"]: raise ValueError("BGE cache changed")
    base=ROOT/"inputs"
    if sha_file(base/"model_reviews.jsonl")!=es["source_input_sha256"]: raise ValueError("Normalized text changed")
    source=[json.loads(x) for x in (base/"model_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    rows=[r for r in source if r["text_clean"].strip()]
    ids=json.loads((Path(es["cache_path"])/"review_ids.json").read_text(encoding="utf-8"))
    vectors=np.load(es["vector_file"],mmap_mode="r",allow_pickle=False);validate_alignment(rows,ids,vectors)
    master=[json.loads(x) for x in (base/"master_index.jsonl").read_text(encoding="utf-8").splitlines()]
    master_ids=[str(r["recommendationid"]) for r in master]
    if not all(isinstance(r["voted_up"],bool) for r in master): raise ValueError("Recommendation diagnostic tags must be explicit booleans")
    votes={str(r["recommendationid"]):bool(r["voted_up"]) for r in master}
    fit=fit_indices(rows)
    if len(fit)!=13145: raise ValueError("Unexpected shared fit scope")
    common=balanced_sample(fit,rows,config["common_sample_size"],config["sampling_seed"])
    evaluation=balanced_sample(common,rows,config["silhouette_sample_size"],config["sampling_seed"]+1)
    output.mkdir(parents=True)
    dump(output/"common_review_order.json",[str(rows[i]["recommendationid"]) for i in common])
    fit_set,common_set,eval_set=set(map(int,fit)),set(map(int,common)),set(map(int,evaluation))
    csv_out(output/"scope_index.csv",[{"recommendationid":str(r["recommendationid"]),"vector_index":i,"used_for_full_fit":i in fit_set,"in_common5000":i in common_set,"in_silhouette1500":i in eval_set,"source_fit_eligible":r["fit_eligible"],"nonlexical":r["nonlexical"],"source_language":r["language"],"writing_system_hint":script_hint(r["text_clean"])} for i,r in enumerate(rows)])
    dump(output/"config.json",config)
    eval_dist=cosine_distances(vectors[evaluation]);np.fill_diagonal(eval_dist,0)
    metrics=[];profiles=[];reps=[];compact={};full_labels={};timings={}
    def save(identifier,alg,scope,k,seed,fi,fl,ai,al,centers=None):
        m,p,r,_=export_candidate(identifier,alg,scope,k,seed,fi,fl,ai,al,vectors,rows,master_ids,votes,evaluation,eval_dist,output,config,centers)
        metrics.append(m);profiles.extend(p);reps.extend(r)
        packed=np.full(len(rows),-1,dtype=int);packed[ai]=al
        compact[identifier]=packed.tolist()
    started=time.perf_counter()
    with threadpool_limits(limits=config["cpu_threads"]):
        for k in config["cluster_counts"]:
            for seed in config["seeds"]:
                t=time.perf_counter();km=KMeans(n_clusters=k,random_state=seed,n_init=config["n_init"],max_iter=config["max_iter"],algorithm="lloyd")
                km.fit(vectors[fit]);applied=km.predict(vectors)
                key=f"full_kmeans_k{k}_s{seed}"
                save(key,"kmeans","full_fit",k,seed,fit,km.labels_,np.arange(len(rows)),applied,km.cluster_centers_)
                full_labels[k,seed]=km.labels_.copy();timings[key]=time.perf_counter()-t
        X=np.asarray(vectors[common])
        for k in config["cluster_counts"]:
            t=time.perf_counter();km=KMeans(n_clusters=k,random_state=config["primary_seed"],n_init=config["n_init"],max_iter=config["max_iter"])
            labels=km.fit_predict(X);key=f"common_kmeans_k{k}_s{config['primary_seed']}"
            save(key,"kmeans","common5000",k,config["primary_seed"],common,labels,common,labels,km.cluster_centers_)
            timings[key]=time.perf_counter()-t
        for method,metric,prefix in [("ward","euclidean","ward"),("average","cosine","average")]:
            t=time.perf_counter();print(f"Building {method}/{metric} tree for {len(common)} shared reviews",flush=True)
            distances=pdist(X,metric=metric)
            tree=linkage(distances,method=method,optimal_ordering=False)
            del distances
            np.save(output/(prefix+"_linkage.npy"),tree,allow_pickle=False)
            cuts=cut_tree(tree,n_clusters=config["cluster_counts"])
            for col,k in enumerate(config["cluster_counts"]):
                labels=cuts[:,col].astype(int)
                save(f"{prefix}_k{k}",method+"_"+metric,"common5000",k,None,common,labels,common,labels)
            timings[prefix+"_tree_and_cuts"]=time.perf_counter()-t
    stability=[{"k":k,"seeds":config["seeds"],"fit_ari":float(adjusted_rand_score(full_labels[k,config["seeds"][0]],full_labels[k,config["seeds"][1]]))} for k in config["cluster_counts"]]
    csv_out(output/"candidate_metrics.csv",metrics);csv_out(output/"cluster_profiles.csv",profiles);csv_out(output/"representative_reviews.csv",reps)
    dump(output/"seed_stability.json",stability)
    summary={"status":"step4_candidates_ready_step5_human_review_pending","created_at":datetime.now(timezone.utc).isoformat(),
             "model":"BAAI/bge-m3","embedding_cache_key":es.get("cache_key",es["vectors_sha256"]),"embedding_vectors_sha256":config["vectors_sha256"],
             "source_commit":config["source_commit"],"encoded_reviews":len(rows),"master_reviews":len(master),"full_fit_reviews":len(fit),
             "common_sample_reviews":len(common),"evaluation_reviews":len(evaluation),"candidate_count":len(metrics),
             "common_source_language_counts":dict(Counter(rows[i]["language"] for i in common)),
             "common_games":len({rows[i]["appid"] for i in common}),"config":config,"seed_stability":stability,
             "compute_seconds":time.perf_counter()-started,"timings":timings,
             "pipeline_sha256":sha_file(Path(__file__)),"encoder_loaded":False,"new_embeddings_computed":0,
             "recommendation_labels_used_for_fitting_or_selection":False,
             "versions":{p:importlib.metadata.version(p) for p in ["numpy","scipy","scikit-learn","threadpoolctl","matplotlib"]},
             "limitations":["API language labels and writing-system rules are not a completed language audit.","Recommendation tags are post-fit diagnostics, not emotion gold or topic-selection targets.","Whole-review hard clusters can mix multiple aspects; no final themes or calibrated no_clear_theme rules exist.","Full-fit and common-sample candidates have different fit scopes; algorithm comparisons use common5000.","Hierarchical labels apply only to its sampled members. No prototype extrapolation to other comments.","Seed stability alone is not resampling or semantic validity. Hierarchical determinism is not evidence of stability."]}
    dump(output/"run_summary.json",summary)
    make_plots(metrics,profiles,output,config);render_report(rows,metrics,profiles,reps,compact,summary,output)
    if sha_file(es["vector_file"])!=config["vectors_sha256"]: raise ValueError("Embedding cache mutated")
    print(json.dumps({"complete":True,"candidates":len(metrics),"seconds":summary["compute_seconds"],"output":str(output)},ensure_ascii=True),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--output",type=Path,default=ROOT/"results/reproduced")
    a=p.parse_args();run(json.loads(a.config.read_text(encoding="utf-8")),a.output)


if __name__=="__main__": main()
