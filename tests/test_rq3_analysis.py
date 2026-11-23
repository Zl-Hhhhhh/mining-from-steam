"""Meaningful RQ3 checks: denominators, pairing, arithmetic and estimability."""
import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rq3_inputs import checked_weights, aggregate_units, common_window_mask, TOPICS
from rq3_statistics import paired_estimate, separation, fit_interaction
from analyze_rq3 import describe_scenario, variants


class RQ3Tests(unittest.TestCase):
    def test_fit_evidence_not_inferred_from_all_scope_flag(self):
        rows = [{'review_id':'1', 'no_topic_evidence':False, 'topic_weights_fit_units':{}, 'topic_weights_all_units':{'T01':1}}]
        self.assertEqual(float(checked_weights(rows,'fit',['1']).sum().sum()),0)
        self.assertEqual(float(checked_weights(rows,'all',['1']).sum().sum()),1)

    def test_duplicate_and_missing_ids_and_invalid_weights_rejected(self):
        row={'review_id':'1','topic_weights_fit_units':{'T01':1}}
        with self.assertRaises(ValueError):checked_weights([row,row],'fit',['1'])
        with self.assertRaises(ValueError):checked_weights([row],'fit',['1','2'])
        for values in [{'T01':.8},{'T01':1.1,'T02':-.1},{'T01':np.nan}]:
            with self.assertRaises(ValueError):checked_weights([{**row,'topic_weights_fit_units':values}],'fit',['1'])

    def test_unit_filter_keeps_other_evidence_and_renormalizes(self):
        labels=[{'unit_id':'a','topic_ids':['T01','T02'],'auxiliary_routes':['A02'],'assignment_scope':'fit_candidate'},
                {'unit_id':'b','topic_ids':['T03'],'auxiliary_routes':[],'assignment_scope':'fit_candidate'}]
        result=aggregate_units(labels,{'1':['a','b'],'2':[]},exclude_routes=['A02'])
        self.assertEqual(result[0]['topic_weights_fit_units'],{'T03':1.})
        self.assertEqual(result[1]['topic_weights_fit_units'],{})

    def test_paired_game_estimator_differs_from_pooled_comments_and_is_order_invariant(self):
        frame=pd.DataFrame({'appid':['1','2'],'english_mean':[.1,.7],'schinese_mean':[.3,.3],
                            'english_n':[100,10],'schinese_n':[10,100]})
        a=paired_estimate(frame,seed=17,reps=1000);b=paired_estimate(frame.iloc[::-1],seed=17,reps=1000)
        self.assertEqual(a,b);self.assertAlmostEqual(a['difference_pp'],-10)
        self.assertGreater(a['comment_weighted_difference_pp'],0)

    def test_common_window_includes_entire_last_utc_day(self):
        dates=pd.to_datetime(['2026-11-01T23:59:59Z','2026-11-02T00:00:00Z'])
        frame=pd.DataFrame({'timestamp_created':[int(x.timestamp()) for x in dates],
                            'common_window_start_utc':['2026-10-31']*2,'common_window_end_utc':['2026-11-01']*2})
        self.assertEqual(common_window_mask(frame).tolist(),[True,False])

    def test_support_checked_within_each_game_and_recomputed_after_selection(self):
        records=[];ws=[]
        for game,n,zh_evidence in [('a',30,30),('b',55,19),('c',55,20)]:
            for lang in ['english','schinese']:
                for i in range(n):
                    records.append({'appid':game,'game':game,'language':lang,'recommend':i%2})
                    ws.append({t:(1. if t=='T01' and i<(zh_evidence if lang=='schinese' else 30) else 0.) for t in TOPICS})
        frame=pd.DataFrame(records);weights=pd.DataFrame(ws)
        cfg={'minimum_reviews_per_language':50,'seed':42,'bootstrap_replicates':200}
        results,_,support,_,cov,theme=describe_scenario('test',frame,weights,20,TOPICS,cfg)
        self.assertEqual(set(support.loc[support.paired_for_coverage,'appid']),{'b','c'})
        self.assertEqual(set(support.loc[support.paired_for_theme,'appid']),{'c'})
        self.assertEqual(len(cov),220);self.assertEqual(len(theme),50)
        self.assertEqual(next(r for r in results if r['measure']=='T01')['status'],'one_game_only')

    def test_complete_separation_is_detected(self):
        x=np.column_stack([np.ones(6),np.array([-3,-2,-1,1,2,3])]);y=np.array([0,0,0,1,1,1])
        self.assertEqual(separation(x,y),'complete')

    def test_fixed_common_window_cohort_does_not_select_by_outcome(self):
        n=240
        frame=pd.DataFrame({'appid':np.repeat(['a','b'],120),'language':np.tile(np.repeat(['english','schinese'],60),2),
                            'association_eligible':True,'in_common_language_window':np.r_[np.ones(120,dtype=bool),np.tile(np.r_[np.ones(20,dtype=bool),np.zeros(40,dtype=bool)],2)],
                            'is_text_canonical':True,'one_per_author':True,'complete_game':True,'is_short_text':False,
                            'script_mismatch':False,'mixed_script':False,'confidence':'low','recommend':np.arange(n)%2})
        w=pd.DataFrame({t:np.full(n,1. if t=='T01' else 0.) for t in TOPICS})
        weights={k:w for k in ['fit','all','no_auxiliary','unrestricted_seeds','k8_overlap_proxy']}
        cfg={'minimum_reviews_per_language':50,'minimum_topic_reviews_per_language':20,'strict_topic_minimum':30}
        a=variants(frame,weights,cfg)['original_same_common_games'][0]
        b=variants(frame.assign(recommend=1-frame.recommend),weights,cfg)['original_same_common_games'][0]
        self.assertEqual(set(frame.loc[a,'appid']),{'a'});self.assertEqual(int(a.sum()),120)
        self.assertTrue(a.equals(b))

    def test_interaction_direct_contrast_equals_chinese_minus_english_log_or(self):
        rng=np.random.default_rng(2504);n=1600
        game=np.repeat(np.arange(10),160).astype(str);zh=np.tile(np.repeat([0,1],80),10)
        weights=pd.DataFrame(rng.dirichlet(np.ones(8),size=n),columns=TOPICS)
        hours=rng.uniform(0,4,n)
        probability=expit(-.4+np.asarray(game,dtype=int)*.04+zh*.1+weights.T03*1.3+zh*weights.T04*3-.08*hours)
        frame=pd.DataFrame({'appid':game,'language':np.where(zh,'schinese','english'),
                            'recommend':rng.binomial(1,probability),'playtime_analysis_eligible':True,'log_hours':hours})
        cfg={'topic_reference':'T01','interaction_minimum_topic_reviews_per_language':10,'interaction_minimum_outcomes_per_language':5}
        with threadpool_limits(limits=2):rows,diag=fit_interaction(frame,weights,cfg)
        self.assertEqual(diag['status'],'ok');self.assertEqual(diag['separation'],'none')
        for topic in TOPICS[1:]:
            selected={r['role']:r for r in rows if r['topic']==topic}
            self.assertAlmostEqual(selected['schinese']['log_or']-selected['english']['log_or'],selected['language_interaction']['log_or'])
            self.assertGreater(selected['language_interaction']['cluster_se'],0)
            self.assertIn('p_holm',selected['language_interaction'])

    def test_rank_deficiency_is_not_reported_as_an_estimable_model(self):
        n=200;frame=pd.DataFrame({'appid':np.repeat(['a','b','c','d'],50),'language':np.tile(['english','schinese'],100),
                                  'recommend':np.tile([0,1,1,0],50),'playtime_analysis_eligible':True,'log_hours':1.})
        weights=pd.DataFrame(np.full((n,8),.125),columns=TOPICS)
        cfg={'topic_reference':'T01','interaction_minimum_topic_reviews_per_language':10,'interaction_minimum_outcomes_per_language':5}
        rows,diag=fit_interaction(frame,weights,cfg)
        self.assertEqual(rows,[]);self.assertEqual(diag['status'],'rank_deficient')


if __name__=='__main__':unittest.main()
