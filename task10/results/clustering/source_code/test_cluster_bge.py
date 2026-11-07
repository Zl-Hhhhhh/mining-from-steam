from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import cluster_bge as c


class ClusteringTests(unittest.TestCase):
    def test_cache_alignment_rejects_permuted_or_duplicate_ids(self):
        rows=[{"recommendationid":"a"},{"recommendationid":"b"}]
        x=np.zeros((2,1024),dtype=np.float32);x[:,0]=1
        c.validate_alignment(rows,["a","b"],x)
        for ids in [["b","a"],["a","a"]]:
            with self.assertRaises(ValueError): c.validate_alignment(rows,ids,x)

    def test_shared_fit_excludes_empty_and_symbols_but_keeps_scope_explicit(self):
        rows=[{"fit_eligible":True,"text_clean":"story","nonlexical":False},
              {"fit_eligible":True,"text_clean":"!","nonlexical":True},
              {"fit_eligible":True,"text_clean":"","nonlexical":True},
              {"fit_eligible":False,"text_clean":"太卡了","nonlexical":False}]
        self.assertEqual(c.fit_indices(rows).tolist(),[0])
        self.assertEqual(len(rows),4)

    def test_shared_sample_is_unique_reproducible_language_balanced(self):
        rows=[{"language":lang,"appid":i%3} for lang in ["english","schinese"] for i in range(20)]
        a=c.balanced_sample(np.arange(40),rows,18,42)
        self.assertEqual(a.tolist(),c.balanced_sample(np.arange(40),rows,18,42).tolist())
        self.assertEqual(len(set(a)),18)
        self.assertEqual(sum(rows[i]["language"]=="english" for i in a),9)

    def test_evaluation_sample_is_subset_of_common_scope(self):
        rows=[{"language":lang,"appid":i%3} for lang in ["english","schinese"] for i in range(30)]
        common=c.balanced_sample(np.arange(60),rows,30,42)
        evaluation=c.balanced_sample(common,rows,10,43)
        self.assertTrue(set(evaluation)<=set(common))

    def test_postfit_diagnostic_labels_do_not_change_fit_selection(self):
        rows=[{"fit_eligible":True,"text_clean":"good story","nonlexical":False,"voted_up":True}]
        changed=[{**rows[0],"voted_up":False,"steamid":"different"}]
        self.assertEqual(c.fit_indices(rows).tolist(),c.fit_indices(changed).tolist())

    def test_script_hint_is_not_claimed_as_actual_english(self):
        self.assertEqual(c.script_hint("これは中文の評論吗"),"Han-dominant")
        self.assertEqual(c.script_hint("un buen juego"),"Latin-script")
        self.assertEqual(c.script_hint("👍"),"other_or_uncertain")


if __name__=="__main__": unittest.main()
