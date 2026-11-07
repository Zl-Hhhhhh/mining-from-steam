"""Synthetic-only encoding control-flow tests; no real-model quality claims."""
import contextlib,hashlib,importlib.util,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
CODE=HERE.parent/'code/task10.py'
spec=importlib.util.spec_from_file_location('portable_task10',CODE);task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')

class PortableTests(unittest.TestCase):
    def fixture(self,root):
        rows=[{'recommendationid':str(i),'text_clean':'TEST synthetic '+str(i)} for i in range(9)]
        rows+=[{'recommendationid':'9','text_clean':rows[0]['text_clean']},{'recommendationid':'empty','text_clean':''}]
        p=root/'inputs/model_reviews.jsonl';p.parent.mkdir();p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
        write(root/'config/embedding.json',{'model_name':'TEST_ONLY_STUB','model_revision':'TEST_PIN','cpu_threads':4,'input_checksums':{'inputs/model_reviews.jsonl':task.sha(p)}})
        write(root/'config/clustering.json',{'vectors_sha256':'before','embedding_summary':'artifacts/embedding_manifest.json'})
        return rows
    def stub_modules(self,fail_second=False,overflow=False):
        calls={'encoded':[],'loads':0,'threads':None,'fail':fail_second}
        class FakeEncoder:
            def __init__(self,*a,**kw):calls['loads']+=1
            def tokenizer(self,text,**kw):return {'input_ids':list(range(8193 if overflow else 20))}
            def encode(self,texts,**kw):
                if calls['fail'] and len(calls['encoded'])>=8:calls['fail']=False;raise RuntimeError('TEST simulated interruption')
                calls['encoded']+=texts
                self_test=kw['normalize_embeddings'] and kw['prompt']=='' and not kw['show_progress_bar']
                if not self_test:raise AssertionError('Encoding settings changed')
                v=np.zeros((len(texts),1024),dtype=np.float32)
                for i,t in enumerate(texts):v[i,int(hashlib.sha256(t.encode()).hexdigest()[:5],16)%1024]=1
                return v
        torch=types.ModuleType('torch');torch.float32='float32';torch.set_num_threads=lambda n:calls.update(threads=n)
        st=types.ModuleType('sentence_transformers');st.SentenceTransformer=FakeEncoder
        hub=types.ModuleType('huggingface_hub');hub.snapshot_download=lambda *a,**kw:'TEST_ONLY_STUB_PATH'
        return {'torch':torch,'sentence_transformers':st,'huggingface_hub':hub},calls
    def test_resume_dedup_ids_fixed_settings_and_cached_skip(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rows=self.fixture(root);modules,calls=self.stub_modules(fail_second=True)
            with patch.object(task,'ROOT',root),patch.dict(sys.modules,modules),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'TEST simulated interruption'):task.encode(offline=True)
                self.assertEqual(len(calls['encoded']),8)
                done=list((root/'.encoding-work').rglob('done.npy'));self.assertEqual(int(np.load(done[0]).sum()),8)
                task.encode(offline=True)
                self.assertEqual(len(calls['encoded']),9);self.assertEqual(len(set(calls['encoded'])),9)
                self.assertEqual(calls['threads'],4)
                ids=json.loads((root/'artifacts/review_ids.json').read_text(encoding='utf-8'));self.assertEqual(ids,[str(i) for i in range(10)])
                v=np.load(root/'artifacts/review_vectors.npy');np.testing.assert_array_equal(v[0],v[9]);self.assertEqual(v.shape,(10,1024))
                self.assertEqual(json.loads((root/'config/clustering-reproduced.json').read_text())['vectors_sha256'],task.sha(root/'artifacts/review_vectors.npy'))
                loads=calls['loads'];task.encode(offline=True);self.assertEqual(calls['loads'],loads)
    def test_no_silent_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root);modules,_=self.stub_modules(overflow=True)
            with patch.object(task,'ROOT',root),patch.dict(sys.modules,modules):
                with self.assertRaisesRegex(ValueError,'No silent truncation'):task.encode(offline=True)
            self.assertFalse((root/'artifacts/review_vectors.npy').exists())
    def test_delivered_real_cache_without_encoder_import(self):
        original_import=__import__
        def guarded(name,*a,**kw):
            if name.split('.')[0] in {'torch','transformers','sentence_transformers','huggingface_hub'}:raise AssertionError('Encoder import attempted')
            return original_import(name,*a,**kw)
        with patch('builtins.__import__',guarded),contextlib.redirect_stdout(io.StringIO()):
            result=task.encode(offline=True)
        self.assertEqual(result['reviews'],19392);self.assertFalse(result['encoder_loaded'])

if __name__=='__main__':unittest.main()
