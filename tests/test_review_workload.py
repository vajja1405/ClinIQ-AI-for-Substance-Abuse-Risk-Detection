import unittest
from analysis.review_workload import estimate_workload

class WorkloadTests(unittest.TestCase):
    def test_lower_prevalence_changes_precision(self):
        x=estimate_workload(120,8,180,292,1000,.1,3)
        self.assertEqual(x['expected_true_alerts'],40)
        self.assertEqual(x['expected_false_alerts'],24)
        self.assertEqual(x['expected_missed_signals'],60)
        self.assertEqual(x['review_hours'],3.2)
        self.assertEqual(x['expected_precision'],.625)
    def test_empty_workload_is_not_perfect_precision(self):
        x=estimate_workload(1,0,0,1,0,.1,3)
        self.assertIsNone(x['expected_precision'])
    def test_invalid_inputs(self):
        for v in [-1, float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError): estimate_workload(1,1,1,1,v,.5,1)
        with self.assertRaises(ValueError): estimate_workload(1,1,1,1,100,1.1,1)
    def test_single_class_evaluation_rejected(self):
        with self.assertRaises(ValueError): estimate_workload(0,1,0,1,100,.1,3)
