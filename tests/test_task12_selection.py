import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_topic_stability as stability
import rework_unsupervised_topics as rework
import rework_nmf_topics as nmf_rework
import prepare_task12_handoff as handoff
import analysis_common as common


class StabilityTests(unittest.TestCase):
    def test_permutation_is_stable_but_a_collapsed_partition_is_not(self):
        reference=np.repeat(np.arange(3),10)
        result=stability.partition_metrics(reference,(reference+1)%3)
        self.assertEqual(result['ari'],1.)
        self.assertEqual(result['macro_cluster_jaccard'],1.)
        collapsed=stability.partition_metrics(reference,np.zeros(30,dtype=int))
        self.assertEqual(collapsed['ari'],0.)
        self.assertLess(collapsed['macro_cluster_jaccard'],.2)

    def test_zero_and_outside_labels_do_not_manufacture_agreement(self):
        result=stability.partition_metrics([-1,0,1],[-1,1,0],[True,False,True])
        self.assertEqual(result['n'],2)
        self.assertEqual(result['excluded_zero_or_outside'],1)
        self.assertEqual(result['out_of_bag_n'],1)
        self.assertIsNone(result['out_of_bag_ari'])
        self.assertIsNone(stability.partition_metrics([-1],[-1])['ari'])

    def test_sampling_is_reproducible_without_duplicates_and_keeps_strata(self):
        strata=['a']*20+['b']*10
        first=stability.stratified_subsample(strata,.8,17)
        self.assertTrue(np.array_equal(first,stability.stratified_subsample(strata,.8,17)))
        self.assertEqual(len(first),24)
        self.assertEqual(len(set(first)),24)
        self.assertEqual(sum(i<20 for i in first),16)

    def test_unsupervised_projection_removes_only_fitted_directions(self):
        X=np.random.default_rng(17).normal(size=(50,8))
        mean,components,transformed=rework.representation_fit_transform(X,2)
        self.assertTrue(np.allclose(transformed@components.T,0,atol=1e-10))
        self.assertTrue(np.allclose(np.linalg.norm(transformed,axis=1),1))
        self.assertTrue(np.allclose(transformed,rework.representation_transform(X,mean,components)))
        self.assertTrue(np.array_equal(rework.representation_fit_transform(X,0)[2],X))

    def test_generic_praise_and_templates_are_not_core_fitting_evidence(self):
        config=common.read_json(common.ROOT/'config/topic_rework.json')
        self.assertFalse(rework.lexical_flags('Very good game!',config)['matches'])
        self.assertIn('performance_stability',rework.lexical_flags('更新后进不去',config)['matches'])
        self.assertTrue(rework.lexical_flags('☐画面差 ☑画面好',config)['template'])
        config={**config,'maximum_core_lexical_categories':1}
        self.assertTrue(rework.lexical_flags('剧情很好，但经常闪退',config)['long_or_many_aspects'])

    def test_sentence_units_preserve_aspect_clauses_and_zero_vectors_stay_unknown(self):
        self.assertEqual(handoff.split_units('剧情很好，但是经常闪退。'),['剧情很好，','经常闪退'])
        self.assertEqual(handoff.split_units('Good story but crashes often!'),['Good story','crashes often'])
        from scipy.sparse import csr_matrix
        labels,weights=nmf_rework.labels_for(np.array([[0.,0.],[1.,0.]]),np.eye(2),csr_matrix([[0.,0.],[1.,0.]]))
        self.assertEqual(labels.tolist(),[-1,0])
        self.assertEqual(weights[0].sum(),0)

    def test_relaxed_handoff_keeps_unmatched_units_for_application_and_routes_generic_text(self):
        config = common.read_json(common.ROOT / "config/topic_rework_residual.json")
        unmatched = "每次进城就只剩十几帧"
        source = [
            {"recommendationid": "1", "appid": 101, "language": "schinese",
             "text_clean": f"好玩。{unmatched}。{unmatched}。剧情很好。"},
            {"recommendationid": "2", "appid": 102, "language": "english",
             "text_clean": f"10/10! Great combat! {unmatched}! 1234!"},
            {"recommendationid": "3", "appid": 103, "language": "english",
             "text_clean": "☐Good story ☑bad story"},
        ]
        master = [*source, {"recommendationid": "4"}]
        baseline, _, baseline_generic = handoff.prepare_units(source, master, config)
        units, review_map, generic = handoff.prepare_units(source, master, config, allow_unmatched=True)

        by_text = {row["text"]: row for row in units.values()}
        self.assertFalse(baseline_generic)
        self.assertEqual({row["text"] for row in baseline.values()}, {"剧情很好", "Great combat"})
        self.assertEqual(set(by_text), {"剧情很好", "Great combat", unmatched})
        self.assertTrue(by_text["Great combat"]["fit_candidate"])
        self.assertEqual(by_text[unmatched]["lexical_categories_diagnostic"], [])
        self.assertFalse(by_text[unmatched]["fit_candidate"])
        self.assertEqual(by_text[unmatched]["review_ids"], ["1", "2"])
        self.assertEqual(review_map["1"].count(by_text[unmatched]["unit_id"]), 1)
        self.assertEqual(review_map["3"], [])
        self.assertEqual(review_map["4"], [])
        self.assertEqual({row["text"] for row in generic.values()}, {"好玩", "10/10"})
        self.assertTrue(all(row["auxiliary_route"] == "A01" for row in generic.values()))


if __name__=='__main__':unittest.main()
