"""Encode the fixed Task12 sentence-unit handoff with BGE-M3, independently of
the historical whole-review cache. No fitting, topic assignment or label input.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,hashlib,html,importlib.metadata,json,os,sys,time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG='config/embedding_bge_sentences.json'
PACKAGES=['torch','transformers','sentence-transformers','numpy','huggingface_hub']
CONTROLS=[
 ('crash','The game keeps crashing.','游戏总是闪退。'),
 ('music','The soundtrack is excellent.','配乐非常出色。'),
 ('story','The story is boring.','剧情很无聊。'),
 ('cheating','There are too many cheaters in matches.','对局里的外挂太多。'),
 ('price','The DLC is too expensive.','这个DLC太贵了。'),
 ('friends','Playing with friends is fun.','和朋友联机很开心。')]

def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def ordered_hash(ids):return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def versions():return {p:importlib.metadata.version(p) for p in PACKAGES}
def peak_memory_gib():
    if os.name!='nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
    try:
        import ctypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        k=ctypes.WinDLL('kernel32');k.GetCurrentProcess.restype=ctypes.c_void_p
        p=ctypes.WinDLL('psapi');p.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        c=Counters();c.cb=ctypes.sizeof(c)
        return c.PeakWorkingSetSize/1024**3 if p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(c),c.cb) else None
    except (OSError,AttributeError):return None
def language_hint(text):
    han=sum('\u3400'<=x<='\u9fff' for x in text);latin=sum(x.isascii() and x.isalpha() for x in text)
    if han>=2 and han>=latin*.4:return 'Han-dominant'
    if latin>=4 and not han:return 'Latin-script'
    return 'other_or_mixed'
def load_inputs(config_path=None):
    c=read_json(config_path or ROOT/DEFAULT_CONFIG)
    request_path=ROOT/c['encoding_request'];request=read_json(request_path)
    if sha_file(request_path)!=c['encoding_request_sha256']:raise ValueError('Encoding request changed')
    for k in ['model_name','model_revision','pooling','normalize_embeddings','dtype','max_seq_length','cpu_threads','max_batch_size']:
        if c[k]!=request[k]:raise ValueError('Encoding request/config mismatch: '+k)
    if not request['encode_all_units'] or request['silent_truncation_allowed']:raise ValueError('Unsupported handoff policy')
    data=ROOT/c['input'];actual=sha_file(data)
    if actual!=request['input_sha256'] or actual!=c['input_sha256']:raise ValueError('Sentence input changed')
    units=[json.loads(x) for x in data.read_text(encoding='utf-8').splitlines()]
    ids=[u['unit_id'] for u in units]
    if any(not isinstance(i,str) or not i for i in ids):raise ValueError('Unit IDs must be nonempty strings')
    if len(ids)!=request['required_shape'][0] or len(set(ids))!=len(ids):raise ValueError('Unit IDs/count are not valid')
    if ordered_hash(ids)!=request['ordered_unit_ids_sha256']:raise ValueError('Unit order changed')
    if request['required_shape'][1]!=1024:raise ValueError('Expected 1024-dimensional BGE dense vectors')
    for u in units:
        if not isinstance(u['text'],str) or not u['text'].strip():raise ValueError('Empty sentence unit')
        if hashlib.sha256(u['text'].encode()).hexdigest()!=u['text_sha256']:raise ValueError('Unit text hash mismatch')
    if len({u['text'] for u in units})!=len(units):raise ValueError('Handoff was expected to contain unique texts')
    summary={'units':len(units),'fit_candidate_units':sum(bool(u['fit_candidate']) for u in units),
             'reviews_with_units':len({rid for u in units for rid in u['review_ids']}),
             'input_sha256':actual,'ordered_unit_ids_sha256':ordered_hash(ids)}
    return c,request,units,summary

def validate_vectors(v,rows):
    import numpy as np
    if v.shape!=(rows,1024) or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('Invalid vector shape/dtype/value')
    if not np.allclose(np.linalg.norm(v,axis=1),1,atol=2e-5):raise ValueError('Non-unit vectors')

def verify(config_path=None):
    import numpy as np
    c,req,units,summary=load_inputs(config_path);out=ROOT/c['output'];m=read_json(out/'sentence_embedding_manifest.json')
    for k in ['model_name','model_revision','pooling','normalize_embeddings','dtype','max_seq_length']:
        if m[k]!=c[k]:raise ValueError('Published cache model/config mismatch: '+k)
    if m['input_sha256']!=summary['input_sha256'] or m['ordered_unit_ids_sha256']!=summary['ordered_unit_ids_sha256']:raise ValueError('Published cache provenance mismatch')
    ids=read_json(out/'unit_ids.json')
    if ids!=[u['unit_id'] for u in units] or m['shape']!=req['required_shape']:raise ValueError('Vector-ID order mismatch')
    vectors=out/'sentence_vectors.npy'
    if sha_file(vectors)!=m['vectors_sha256']:raise ValueError('Published vector checksum mismatch')
    if sha_file(out/'unit_ids.json')!=m['unit_ids_file_sha256']:raise ValueError('Published ID checksum mismatch')
    v=np.load(vectors,mmap_mode='r',allow_pickle=False)
    if list(v.shape)!=req['required_shape']:raise ValueError('Published vector count mismatch')
    for i in range(0,len(v),2048):validate_vectors(v[i:i+2048],len(v[i:i+2048]))
    result={'status':'passed','phase':'cache_verification','shape':list(v.shape),'input_and_id_order_verified':True,'vectors_sha256':m['vectors_sha256'],'encoder_loaded_by_verifier':False}
    print(json.dumps(result),flush=True);return result

def load_encoder(c,model_snapshot=None):
    os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    from sentence_transformers import SentenceTransformer
    import torch
    if model_snapshot is None:
        from huggingface_hub import snapshot_download
        model_snapshot=Path(snapshot_download(c['model_name'],revision=c['model_revision'],cache_dir=ROOT/c['model_cache_dir'],local_files_only=True,token=False))
    model_snapshot=Path(model_snapshot).resolve()
    if model_snapshot.name!=c['model_revision']:raise ValueError('Snapshot folder must identify the pinned model revision')
    started=time.perf_counter()
    for rel,expected in c['model_files_sha256'].items():
        if sha_file(model_snapshot/rel)!=expected:raise ValueError('Pinned model file changed: '+rel)
    modules=read_json(model_snapshot/'modules.json')
    if [x['type'] for x in modules]!=['sentence_transformers.models.Transformer','sentence_transformers.models.Pooling','sentence_transformers.models.Normalize']:raise ValueError('Unexpected model modules')
    torch.set_num_threads(c['cpu_threads'])
    model=SentenceTransformer(str(model_snapshot),device='cpu',local_files_only=True,trust_remote_code=False,
        model_kwargs={'dtype':torch.float32,'attn_implementation':'sdpa','weights_only':True})
    model.max_seq_length=c['max_seq_length'];pool=model[1]
    mode=getattr(pool,'pooling_mode',None)
    if mode is not None:
        modes=(mode,) if isinstance(mode,str) else tuple(mode)
        if modes!=('cls',):raise ValueError('CLS pooling was not preserved')
    elif not getattr(pool,'pooling_mode_cls_token',False) or any(getattr(pool,k,False) for k in ['pooling_mode_mean_tokens','pooling_mode_max_tokens','pooling_mode_mean_sqrt_len_tokens','pooling_mode_weightedmean_tokens','pooling_mode_lasttoken']):raise ValueError('CLS pooling was not preserved')
    print(json.dumps({'phase':'encoder_ready','model_files_verified':len(c['model_files_sha256']),'cpu_threads':c['cpu_threads'],'seconds':round(time.perf_counter()-started,2)}),flush=True)
    return model

def make_batches(indices,lengths,max_batch,max_padded):
    batch=[]
    for i in sorted(indices,key=lambda i:(lengths[i],int(i))):
        if batch and (len(batch)>=max_batch or lengths[i]*(len(batch)+1)>max_padded):yield batch;batch=[]
        batch.append(int(i))
    if batch:yield batch

def checkpoint_key(c,summary):
    policy={k:c[k] for k in ['model_name','model_revision','pooling','normalize_embeddings','dtype','max_seq_length','cpu_threads','max_batch_size','max_padded_tokens_per_batch']}
    return digest({'input':summary,'policy':policy,'versions':versions(),'model_files_sha256':c['model_files_sha256'],'encoder_script_sha256':sha_file(__file__)})

def open_checkpoint(work,key,n):
    import numpy as np
    meta=work/'checkpoint.json';vp=work/'vectors.npy';dp=work/'completed.npy'
    if meta.exists():
        if read_json(meta)!={'fingerprint':key,'shape':[n,1024]}:raise ValueError('Checkpoint fingerprint changed')
    else:
        if vp.exists() or dp.exists():raise ValueError('Interrupted checkpoint initialization; preserve files and choose a new work directory')
        a=np.lib.format.open_memmap(vp,mode='w+',dtype=np.float32,shape=(n,1024));a[:]=np.nan;a.flush();del a
        d=np.lib.format.open_memmap(dp,mode='w+',dtype=bool,shape=(n,));d[:]=False;d.flush();del d
        write_json(meta,{'fingerprint':key,'shape':[n,1024]})
    vectors=np.load(vp,mmap_mode='r+',allow_pickle=False);done=np.load(dp,mmap_mode='r+',allow_pickle=False)
    if vectors.shape!=(n,1024) or vectors.dtype!=np.float32 or done.shape!=(n,) or done.dtype!=bool:raise ValueError('Checkpoint array mismatch')
    for start in range(0,n,2048):
        selected=np.flatnonzero(done[start:start+2048])+start
        if len(selected):validate_vectors(vectors[selected],len(selected))
    return vectors,done

def choose_pilot(units,n=48):
    n=min(n,len(units));chosen=[]
    def add(i):
        if i not in chosen and len(chosen)<n:chosen.append(i)
    for lang in ['Latin-script','Han-dominant']:
        group=[i for i,u in enumerate(units) if language_hint(u['text'])==lang]
        ranked=sorted(group,key=lambda i:(len(units[i]['text']),units[i]['unit_id']))
        for i in ranked[:3]+ranked[-3:]:add(i)
        for fit in [True,False]:
            pool=[i for i in group if bool(units[i]['fit_candidate'])==fit]
            for i in sorted(pool,key=lambda i:digest(['pilot-20261117',units[i]['unit_id']]))[:6]:add(i)
    for i in sorted(range(len(units)),key=lambda i:digest(['fill',units[i]['unit_id']])):add(i)
    return sorted(chosen)

def fill_vectors(model,units,lengths,indices,c,vectors,done):
    import numpy as np
    if max(lengths,default=0)>c['max_seq_length']:raise ValueError('No silent truncation: sentence exceeds native input limit')
    pending=[i for i in indices if not done[i]];batches=list(make_batches(pending,lengths,c['max_batch_size'],c['max_padded_tokens_per_batch']))
    start=time.perf_counter()
    for k,batch in enumerate(batches,1):
        v=model.encode([units[i]['text'] for i in batch],prompt='',batch_size=len(batch),show_progress_bar=False,normalize_embeddings=True,convert_to_numpy=True)
        v=np.asarray(v,dtype=np.float32);validate_vectors(v,len(batch))
        vectors[batch]=v;vectors.flush();done[batch]=True;done.flush()
        if k%10==0 or k==len(batches):print(json.dumps({'phase':'encoding','batch':k,'batches':len(batches),'completed':int(done.sum()),'total':len(units),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    return {'newly_encoded':len(pending),'encoding_seconds':time.perf_counter()-start,'batches':len(batches)}

def write_pilot(model,units,indices,vectors,lengths,out,key,summary,c):
    import numpy as np
    out=out/'pilot';out.mkdir(parents=True,exist_ok=True)
    if (out/'pilot_summary.json').exists() and read_json(out/'pilot_summary.json')['fingerprint']!=key:raise ValueError('Pilot from another encoder version exists; use a new output directory')
    pv=np.asarray(vectors[indices]);validate_vectors(pv,len(indices));np.save(out/'pilot_vectors.npy',pv,allow_pickle=False)
    write_json(out/'pilot_unit_ids.json',[units[i]['unit_id'] for i in indices])
    control_texts=[t for _,en,zh in CONTROLS for t in [en,zh]]
    cv=model.encode(control_texts,prompt='',batch_size=c['max_batch_size'],show_progress_bar=False,normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
    validate_vectors(cv,len(control_texts));np.save(out/'synthetic_control_vectors.npy',cv,allow_pickle=False)
    cross=cv[::2]@cv[1::2].T;probes=[]
    for i,(name,en,zh) in enumerate(CONTROLS):
        order=np.argsort(-cross[i]);probes.append({'pair':name,'english':en,'chinese':zh,'paired_cosine':float(cross[i,i]),'paired_rank_among_six_chinese':int(np.flatnonzero(order==i)[0])+1,'closest_chinese':CONTROLS[int(order[0])][2]})
    sim=pv@pv.T;examples=[]
    for q,i in enumerate(indices):
        lang=language_hint(units[i]['text']);opposite='Han-dominant' if lang=='Latin-script' else 'Latin-script'
        available=[j for j,idx in enumerate(indices) if j!=q and language_hint(units[idx]['text'])==opposite]
        closest=sorted(available,key=lambda j:-float(sim[q,j]))[:3]
        examples.append({'unit_id':units[i]['unit_id'],'text':units[i]['text'],'language_hint':lang,'fit_candidate':units[i]['fit_candidate'],'cross_language_neighbors':[{'unit_id':units[indices[j]]['unit_id'],'text':units[indices[j]]['text'],'cosine':float(sim[q,j])} for j in closest]})
    report={'status':'pilot_runtime_checks_passed_semantic_examples_for_review','fingerprint':key,'real_units':len(indices),'language_hints':dict(Counter(language_hint(units[i]['text']) for i in indices)),'fit_candidate_counts':dict(Counter(str(units[i]['fit_candidate']) for i in indices)),'max_tokens_in_all_input':max(lengths),'truncated_units':0,'synthetic_controls':probes,'real_unit_neighbors':examples,'input':summary,'peak_process_memory_gib':peak_memory_gib(),'interpretation':'Control pairs are constructed sanity probes, not independent gold accuracy. Real neighbors are limited to the pilot subset, not a full-corpus retrieval result. No topic labels assigned.'}
    write_json(out/'pilot_summary.json',report)
    e=lambda x:html.escape(str(x));blocks=[]
    for x in examples:
        neighbors=''.join('<li><strong>'+e(round(y['cosine'],4))+'</strong> · '+e(y['unit_id'])+'<p>'+e(y['text'])+'</p></li>' for y in x['cross_language_neighbors'])
        blocks.append('<details class="card"><summary>'+e(x['unit_id'])+' · '+e(x['language_hint'])+'</summary><p>'+e(x['text'])+'</p><ol>'+neighbors+'</ol></details>')
    table=''.join('<tr><td>'+e(p['english'])+'</td><td>'+e(p['chinese'])+'</td><td>'+e(round(p['paired_cosine'],4))+'</td><td>'+e(p['paired_rank_among_six_chinese'])+'</td></tr>' for p in probes)
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BGE句子编码小批量检查</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f2f5f8;color:#172638;margin:0}main{max-width:1100px;margin:auto;padding:24px}.card{background:white;padding:18px;border:1px solid #d9e1e8;border-radius:8px;margin:12px 0}p,td{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd5de;padding:9px;text-align:left}.note{background:#fff3d1;padding:16px}summary{cursor:pointer}@media(max-width:700px){main{padding:10px}table{font-size:13px}}</style><main><h1>BGE句子编码：小批量检查</h1><p>真实输入48个单元＋6对人为构造的中英文对照句。固定BGE-M3修订，CLS、L2、float32。</p><p class="note">这里检查运行、ID映射和近邻阅读；不是正式主题准确率。真实近邻仅在48条小样本内寻找，未给评论分配主题。对照句独立于正式单元，不混入全量缓存。</p><h2>构造对照句</h2><table><tr><th>英文</th><th>中文</th><th>余弦相似度</th><th>6个中文中的配对排名</th></tr>'''+table+'</table><h2>真实单元的跨书写系统近邻</h2>'+''.join(blocks)+'</main>'
    (out/'report.html').write_text(page,encoding='utf-8');return report

def publish(c,req,units,summary,vectors,done,lengths,key,stats):
    import numpy as np
    if not done.all():raise ValueError('Refuse to publish an incomplete checkpoint')
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    marker=out/'.publishing.json'
    if any((out/n).exists() for n in ['sentence_vectors.npy','unit_ids.json']) and (not marker.exists() or read_json(marker).get('fingerprint')!=key):raise ValueError('Unidentified partial output exists; preserve it and choose another output version')
    write_json(marker,{'fingerprint':key,'input_sha256':summary['input_sha256']})
    tmp=out/'sentence_vectors.assembling.npy';v=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(units),1024))
    for i in range(0,len(units),2048):v[i:i+2048]=vectors[i:i+2048];validate_vectors(v[i:i+2048],len(v[i:i+2048]))
    v.flush();del v;tmp.replace(out/'sentence_vectors.npy')
    write_json(out/'unit_ids.json',[u['unit_id'] for u in units])
    m={k:c[k] for k in ['model_name','model_revision','pooling','normalize_embeddings','dtype','max_seq_length','cpu_threads','max_batch_size','max_padded_tokens_per_batch']}
    m.update(schema_version='1.0',vector_file='sentence_vectors.npy',unit_ids_file='unit_ids.json',shape=req['required_shape'],
        input_path=c['input'],input_sha256=summary['input_sha256'],ordered_unit_ids_sha256=summary['ordered_unit_ids_sha256'],
        vectors_sha256=sha_file(out/'sentence_vectors.npy'),unit_ids_file_sha256=sha_file(out/'unit_ids.json'),
        fit_candidate_units=summary['fit_candidate_units'],reviews_with_units=summary['reviews_with_units'],
        source_commit=c['source_commit'],encoding_request_sha256=c['encoding_request_sha256'],
        model_files_sha256=c['model_files_sha256'],runtime_versions=versions(),generator_sha256=sha_file(__file__),
        checkpoint_fingerprint=key,max_observed_tokens=int(max(lengths)),units_over512_tokens=sum(x>512 for x in lengths),
        truncated_units=0,old_whole_review_vectors_reused=False,topic_labels_used_for_encoding=False,
        recommendation_labels_used=False,topic_assignment_performed=False,run_statistics=stats)
    # The manifest is the last publication marker. Missing marker => resume from
    # checkpoint and reassemble; a present but invalid marker => fail, never overwrite.
    write_json(out/'sentence_embedding_manifest.json',m)
    marker.unlink()
    return m

def run(action,config_path=None,model_snapshot=None,pilot_size=48,encoder_factory=None):
    c,req,units,summary=load_inputs(config_path);out=ROOT/c['output']
    if action=='preflight':print(json.dumps({'status':'passed',**summary,'required_shape':req['required_shape']}));return summary
    if action=='verify':return verify(config_path)
    if action=='encode' and (out/'sentence_embedding_manifest.json').exists():return verify(config_path)
    from filelock import FileLock
    import numpy as np
    key=checkpoint_key(c,summary);work=ROOT/c['checkpoint_dir']/key;work.mkdir(parents=True,exist_ok=True)
    with FileLock(str(work/'run.lock'),timeout=0):
        start=time.perf_counter();model=(encoder_factory or load_encoder)(c,model_snapshot)
        lengths=[len(model.tokenizer(u['text'],truncation=False)['input_ids']) for u in units]
        if max(lengths)>c['max_seq_length']:raise ValueError('No silent truncation: sentence exceeds native input limit')
        print(json.dumps({'phase':'token_lengths_checked','units':len(units),'max_tokens':max(lengths),'over512':sum(x>512 for x in lengths),'truncated':0}),flush=True)
        vectors,done=open_checkpoint(work,key,len(units));before=int(done.sum())
        indices=choose_pilot(units,pilot_size) if action=='pilot' else list(range(len(units)))
        stats=fill_vectors(model,units,lengths,indices,c,vectors,done)
        stats.update(cache_units_reused_at_start=before,elapsed_seconds_this_process=time.perf_counter()-start,peak_process_memory_gib=peak_memory_gib())
        if action=='pilot':
            report=write_pilot(model,units,indices,vectors,lengths,out,key,summary,c)
            print(json.dumps({'phase':'pilot_complete','units':len(indices),'synthetic_pair_ranks':[x['paired_rank_among_six_chinese'] for x in report['synthetic_controls']],'report':str(out/'pilot/report.html')}),flush=True)
            return report
        manifest=publish(c,req,units,summary,vectors,done,lengths,key,stats)
        print(json.dumps({'phase':'complete','shape':manifest['shape'],'newly_encoded':stats['newly_encoded'],'cache_units_reused':before,'seconds':round(stats['elapsed_seconds_this_process'],1)}),flush=True)
        del vectors,done,model
    return verify(config_path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','pilot','encode','verify']);p.add_argument('--config',type=Path);p.add_argument('--model-snapshot',type=Path);p.add_argument('--pilot-size',type=int,default=48)
    a=p.parse_args();run(a.action,a.config,a.model_snapshot,a.pilot_size)
if __name__=='__main__':main()
