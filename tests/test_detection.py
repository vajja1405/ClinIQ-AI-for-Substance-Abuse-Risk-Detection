"""Exercise shared application logic; no duplicated implementation or live services."""
from pathlib import Path
import pytest
from analysis.detection import rule_classify
from analysis.evidence import metrics_from_counts,load_saved_comparison,opportunity_funnel,claim_payment_delta
ROOT=Path(__file__).resolve().parents[1]

def test_rule_baseline_and_known_substring_limitation():
 keywords={'opioid':['opioid','heroin']}
 assert rule_classify('OPIOID therapy','',keywords)==(1,['opioid'])
 assert rule_classify('helped my pain','opioid',keywords)==(1,['opioid'])
 assert rule_classify('heroine of the story','',keywords)==(1,['opioid'])
 assert rule_classify('','',keywords)==(0,[])

def test_saved_comparison_uses_counts_not_rounded_claims():
 rows=load_saved_comparison(ROOT/'outputs/method_comparison_results.csv')
 llm=next(r for r in rows if r['method']=='llm_rag')
 assert llm['precision']==pytest.approx(120/128)
 assert llm['recall']==pytest.approx(120/300)
 assert all(r['n']==600 and 'proxy' in r['label_basis'] for r in rows)

def test_undefined_precision_not_zero_performance():
 assert metrics_from_counts(0,0,1,9)['precision'] is None
 for counts in [(0,0,0,0),(-1,1,0,0),(True,1,0,0)]:
  with pytest.raises(ValueError):metrics_from_counts(*counts)

def test_funnel_conditions_and_costs():
 args=dict(admissions=10000,candidate_share=.05,missed_share=.4,valid_share=.6,payment_change_share=.5,realization_share=1,incremental_payment=3000,review_cost_per_candidate=30)
 r=opportunity_funnel(**args)
 assert r['gross']==180000 and r['net']==165000
 assert opportunity_funnel(**{**args,'payment_change_share':0})['net']==-15000
 for key,val in [('valid_share',2),('implementation_cost',float('nan')),('admissions',-1)]:
  with pytest.raises(ValueError):opportunity_funnel(**{**args,key:val})

def test_unvalidated_coding_changes_cannot_generate_revenue():
 assert claim_payment_delta() is None
 assert claim_payment_delta(10,20) is None
 assert claim_payment_delta(20,10,True)==-10
