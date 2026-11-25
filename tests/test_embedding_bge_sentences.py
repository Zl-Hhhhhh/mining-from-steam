"""Synthetic-only safety tests for the sentence contract, caching and row order."""
import contextlib,hashlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
import numpy as np
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sentence_encoder',R/'scripts/embedding_bge_sentences.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(root):
    units=[]
    for i in range(10):
        text='TEST synthetic unit '+str(i)+(' x'*(10-i))
        units.append({'unit_id':f'TEST{i:02}','text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'review_ids':[str(i//2)],'fit_candidate':i%2==0})
    p=root/'handoff/units.jsonl';p.parent.mkdir(parents=True);p.write_text('\n'.join(json.dumps(u) for u in units)+'\n',encoding='utf-8')
    request={'encode_all_units':True,'silent_truncation_allowed':False,'model_name':'TEST_ONLY','model_revision':'TEST_REV','pooling':'cls','normalize_embeddings':True,'dtype':'float32','max_seq_length':8192,'cpu_threads':2,'max_batch_size':2,'input_sha256':m.sha_file(p),'required_shape':[10,1024],'ordered_unit_ids_sha256':m.ordered_hash([u['unit_id'] for u in units])}
    m.write_json(root/'handoff/request.json',request)
    c={**request,'input':'handoff/units.jsonl','encoding_request':'handoff/request.json','encoding_request_sha256':m.sha_file(root/'handoff/request.json'),'output':'outputs/new-sentences','checkpoint_dir':'work','model_cache_dir':'model-cache','max_padded_tokens_per_batch':4096,'model_files_sha256':{},'source_commit':'TEST_ONLY'}
    cp=root/'config/embedding_bge_sentences.json';m.write_json(cp,c);return cp,c,units

class Fake:
    def __init__(self):self.calls=[];self.fail_on=None;self.overflow=False
    def tokenizer(self,t,**kw):return {'input_ids':list(range(8193 if self.overflow else len(t)))}
    def encode(self,texts,**kw):
        if self.fail_on is not None and len(self.calls)>=self.fail_on:self.fail_on=None;raise RuntimeError('TEST interruption')
        self.calls+=texts
        v=np.zeros((len(texts),1024),dtype=np.float32)
        for i,t in enumerate(texts):v[i,int(hashlib.sha256(t.encode()).hexdigest()[:6],16)%1024]=1
        return v

class SentenceTests(unittest.TestCase):
    def test_offline_model_lookup_is_limited_to_the_pinned_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot = root / "fixed-revision"
            m.write_json(snapshot / "modules.json", [
                {"type": f"sentence_transformers.models.{name}"}
                for name in ("Transformer", "Pooling", "Normalize")
            ])
            m.write_json(snapshot / "1_Pooling/config.json", {"pooling_mode_cls_token": True})
            files = {name: m.sha_file(snapshot / name) for name in ("modules.json", "1_Pooling/config.json")}
            config = {
                "model_name": "TEST/model", "model_revision": snapshot.name, "model_cache_dir": "cache",
                "model_files_sha256": files, "cpu_threads": 2, "max_seq_length": 8192,
            }
            lookup = Mock(return_value=str(snapshot))
            model = MagicMock()
            model[1].pooling_mode = "cls"
            modules = {
                "huggingface_hub": SimpleNamespace(snapshot_download=lookup),
                "sentence_transformers": SimpleNamespace(SentenceTransformer=Mock(return_value=model)),
                "torch": SimpleNamespace(set_num_threads=Mock(), float32=object()),
            }
            with patch.object(m, "ROOT", root), patch.dict(sys.modules, modules), patch.dict(m.os.environ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIs(m.load_encoder(config), model)
            lookup.assert_called_once_with("TEST/model", revision=snapshot.name,
                                           cache_dir=root / "cache", allow_patterns=list(files),
                                           local_files_only=True, token=False)

    def test_resume_avoids_reencoding_and_preserves_source_id_order(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,units=fixture(root);fake=Fake();fake.fail_on=2
            with patch.object(m,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'TEST interruption'):m.run('encode',cp,encoder_factory=lambda *_:fake)
                self.assertEqual(len(fake.calls),2)
                m.run('encode',cp,encoder_factory=lambda *_:fake)
                self.assertEqual(len(fake.calls),10);self.assertEqual(len(set(fake.calls)),10)
                self.assertEqual(m.read_json(root/c['output']/'unit_ids.json'),[u['unit_id'] for u in units])
                actual=np.load(root/c['output']/'sentence_vectors.npy');expected=Fake().encode([u['text'] for u in units]);np.testing.assert_array_equal(actual,expected)
                self.assertFalse((root/c['output']/'.publishing.json').exists())
                # A complete cache requires no encoder or heavyweight imports.
                def fail_loader(*a):raise AssertionError('Encoder should not load')
                original=__import__
                def guard(name,*a,**kw):
                    if name.split('.')[0] in {'torch','transformers','sentence_transformers','huggingface_hub'}:raise AssertionError(name)
                    return original(name,*a,**kw)
                with patch('builtins.__import__',guard):result=m.run('encode',cp,encoder_factory=fail_loader)
                self.assertFalse(result['encoder_loaded_by_verifier'])
    def test_partial_pilot_checkpoint_reused_for_full_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,units=fixture(root);fake=Fake()
            with patch.object(m,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                _,_,_,summary=m.load_inputs(cp);key=m.checkpoint_key(c,summary);work=root/c['checkpoint_dir']/key;work.mkdir(parents=True)
                v,d=m.open_checkpoint(work,key,10);lengths=[len(u['text']) for u in units]
                m.fill_vectors(fake,units,lengths,[1,3,7,8],c,v,d);del v,d
                m.run('encode',cp,encoder_factory=lambda *_:fake)
                self.assertEqual(len(fake.calls),10)
                manifest=m.read_json(root/c['output']/'sentence_embedding_manifest.json')
                self.assertEqual(manifest['run_statistics']['cache_units_reused_at_start'],4)
                self.assertEqual(manifest['run_statistics']['newly_encoded'],6)
    def test_input_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,_=fixture(root)
            with (root/c['input']).open('a',encoding='utf-8') as f:f.write(' ')
            with patch.object(m,'ROOT',root),self.assertRaisesRegex(ValueError,'input changed'):m.load_inputs(cp)
    def test_id_order_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,units=fixture(root);units.reverse()
            p=root/c['input'];p.write_text('\n'.join(json.dumps(u) for u in units)+'\n',encoding='utf-8')
            req=m.read_json(root/c['encoding_request']);req['input_sha256']=m.sha_file(p);m.write_json(root/c['encoding_request'],req)
            c['input_sha256']=req['input_sha256'];c['encoding_request_sha256']=m.sha_file(root/c['encoding_request']);m.write_json(cp,c)
            with patch.object(m,'ROOT',root),self.assertRaisesRegex(ValueError,'order changed'):m.load_inputs(cp)
    def test_duplicate_unit_ids_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,units=fixture(root);units[1]['unit_id']=units[0]['unit_id']
            p=root/c['input'];p.write_text('\n'.join(json.dumps(u) for u in units)+'\n',encoding='utf-8')
            req=m.read_json(root/c['encoding_request']);req['input_sha256']=m.sha_file(p);m.write_json(root/c['encoding_request'],req)
            c['input_sha256']=req['input_sha256'];c['encoding_request_sha256']=m.sha_file(root/c['encoding_request']);m.write_json(cp,c)
            with patch.object(m,'ROOT',root),self.assertRaisesRegex(ValueError,'IDs/count'):m.load_inputs(cp)
    def test_no_silent_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,_=fixture(root);fake=Fake();fake.overflow=True
            with patch.object(m,'ROOT',root),self.assertRaisesRegex(ValueError,'No silent truncation'):m.run('encode',cp,encoder_factory=lambda *_:fake)
            self.assertFalse(fake.calls);self.assertFalse((root/c['output']/'sentence_embedding_manifest.json').exists())
    def test_padded_budget_and_batch_cap(self):
        batches=list(m.make_batches([0,1,2,3],[30,2000,1000,2500],2,4096))
        self.assertEqual([x for b in batches for x in b],[0,2,1,3])
        for b in batches:self.assertLessEqual(len(b),2);self.assertTrue(len(b)==1 or max([30,2000,1000,2500][i] for i in b)*len(b)<=4096)
    def test_unrelated_partial_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,_=fixture(root);out=root/c['output'];out.mkdir(parents=True);p=out/'sentence_vectors.npy';p.write_bytes(b'TEST preserve this')
            with patch.object(m,'ROOT',root),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'Unidentified partial'):m.run('encode',cp,encoder_factory=lambda *_:Fake())
            self.assertEqual(p.read_bytes(),b'TEST preserve this')
    def test_cache_corruption_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cp,c,_=fixture(root)
            with patch.object(m,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):m.run('encode',cp,encoder_factory=lambda *_:Fake())
            p=root/c['output']/'sentence_vectors.npy'
            with p.open('r+b') as f:f.seek(-1,2);f.write(b'\x7f')
            with patch.object(m,'ROOT',root),self.assertRaisesRegex(ValueError,'checksum'):m.verify(cp)

if __name__=='__main__':unittest.main()
