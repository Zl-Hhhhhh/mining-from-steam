import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import embedding_bge as bge


class EncodingTests(unittest.TestCase):
    def test_exact_text_reuse_preserves_case_and_review_order(self):
        rows = [{"text_clean":t} for t in ["太卡了", "Great game", "太卡了", "great game"]]
        texts, indices = bge.unique_text_plan(rows)
        self.assertEqual(texts, ["太卡了", "Great game", "great game"])
        self.assertEqual(indices, [0,1,0,2])
        self.assertEqual([texts[i] for i in indices], bge.encoder_texts(rows))

    def test_empty_reviews_cannot_receive_a_fake_vector(self):
        with self.assertRaises(ValueError):
            bge.unique_text_plan([{"text_clean":" "}])

    def test_labels_game_and_language_are_not_features(self):
        a = [{"text_clean":"全是挂", "recommendationid":"1", "voted_up":False, "language":"schinese"}]
        c = [{"text_clean":"全是挂", "recommendationid":"2", "voted_up":True, "language":"english", "appid":3}]
        self.assertEqual(bge.encoder_texts(a), bge.encoder_texts(c))

    def test_dynamic_batches_cover_each_input_once_and_stay_in_budget(self):
        lengths = [4,8,9,512,800,3993,2100,50,60,5,17]
        batches = list(bge.batch_indices(list(range(len(lengths))), lengths, 3, 1024))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(lengths))))
        self.assertTrue(all(len(batch)<=3 for batch in batches))
        self.assertTrue(all(len(batch)==1 or max(lengths[i] for i in batch)*len(batch)<=1024 for batch in batches))

    def test_lengths_over_batch_budget_are_alone_not_truncated(self):
        self.assertEqual(list(bge.batch_indices([0,1,2], [3993,5000,15],8,4096)), [[2],[0],[1]])

    def test_vectors_reject_nans_zero_norm_wrong_dimension(self):
        good = np.zeros((2,1024),dtype=np.float32);good[:,0]=1
        bge.validate_vectors(good,2,1024)
        for bad in [np.zeros_like(good), good[:,:10], np.full_like(good,np.nan)]:
            with self.assertRaises(ValueError):
                bge.validate_vectors(bad,2,1024)

    def test_identity_changes_for_text_order_or_model_revision(self):
        x={"ids":["1","2"],"texts":["A","B"],"revision":"r1"}
        self.assertNotEqual(bge.digest(x),bge.digest({**x,"ids":["2","1"]}))
        self.assertNotEqual(bge.digest(x),bge.digest({**x,"revision":"r2"}))


if __name__ == "__main__":
    unittest.main()
