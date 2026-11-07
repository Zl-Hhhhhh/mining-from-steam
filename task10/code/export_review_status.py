"""Regenerate conservative review-level status flags from delivered assignments.

These flags are reading priorities / provisional rejections, not human themes
or calibrated confidence probabilities. Original candidate assignments remain.
"""
import argparse,csv,json,re,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['recommendationid','raw_cluster','assignment_status','generic_only_rule_flag','boundary_flag','human_aspect_presence','human_theme_labels','confidence_probability','note']
def main():
    p=argparse.ArgumentParser();p.add_argument('--assignments',type=Path,default=ROOT/'results/clustering/full_kmeans_k10_s17/assignments.csv');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a new output file; existing evidence is never overwritten')
    rules=json.loads((ROOT/'provisional_status_rules.json').read_text(encoding='utf-8'));generic=re.compile(rules['generic_fullmatch_regex'])
    text_path=ROOT/'inputs/model_reviews.jsonl';manifest=json.loads((ROOT/'artifacts/embedding_manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(text_path.read_bytes()).hexdigest()!=manifest['source_input_sha256']:raise ValueError('Normalized frozen input changed')
    rows={str(r['recommendationid']):r for r in map(json.loads,text_path.read_text(encoding='utf-8').splitlines())}
    with a.assignments.open(encoding='utf-8-sig',newline='') as f:assignments=list(csv.DictReader(f))
    master_ids=[str(r['recommendationid']) for r in map(json.loads,(ROOT/'inputs/master_index.jsonl').read_text(encoding='utf-8').splitlines())]
    if [x['recommendationid'] for x in assignments]!=master_ids or any(x['raw_cluster']=='-1' for x in assignments if x['recommendationid'] in rows and rows[x['recommendationid']]['text_clean'].strip()):raise ValueError('Expected full-coverage KMeans assignments; do not extrapolate hierarchy labels')
    out=[]
    for x in assignments:
        rid=x['recommendationid'];r=rows.get(rid);text=r['text_clean'] if r else ''
        if not text.strip():status='empty_text'
        elif r['nonlexical']:status='provisional_no_clear_theme_nonlexical'
        elif generic.fullmatch(text.strip()):status='provisional_no_clear_theme_generic_rule'
        elif x['boundary_reading_flag']=='True':status='low_confidence_review_required'
        else:status='unvalidated_candidate'
        out.append(dict(zip(FIELDS,[rid,x['raw_cluster'],status,bool(text and generic.fullmatch(text.strip())),x['boundary_reading_flag']=='True','','','','Rules and center gap are uncalibrated; not human gold'])))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(out)
    print(json.dumps({'rows':len(out),'status_counts':dict(Counter(r['assignment_status'] for r in out)),'human_labels_generated':False}))
if __name__=='__main__':main()
