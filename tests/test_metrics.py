import unittest
import numpy as np
from measure import summary_ms
from workload import PAIRS, corpus, queries, performance_texts, ranking_metrics, workload_hash


class MetricTests(unittest.TestCase):
    def test_nearest_rank_p95(self):
        result = summary_ms(list(range(1, 21)))
        self.assertEqual(result['p95_ms'], 19)
        self.assertEqual(result['p50_ms'], 10.5)

    def test_empty_measurement_is_not_zero_latency(self):
        self.assertIsNone(summary_ms([])['p95_ms'])

    def test_exact_retrieval(self):
        result = ranking_metrics(np.eye(len(PAIRS)))
        self.assertEqual(result['recall_at_1'], 1)
        self.assertEqual(result['mrr'], 1)

    def test_ties_preserve_document_order(self):
        result = ranking_metrics(np.ones((len(PAIRS), len(PAIRS))))
        self.assertEqual(result['recall_at_1'], 1 / len(PAIRS))
        self.assertAlmostEqual(result['mrr'], sum(1 / r for r in range(1, len(PAIRS) + 1)) / len(PAIRS))

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            ranking_metrics(np.full((len(PAIRS), len(PAIRS)), np.nan))

    def test_authored_references_are_unique_and_complete(self):
        self.assertEqual(len({r['id'] for r in corpus()}), len(PAIRS))
        self.assertEqual({r['relevant_id'] for r in queries()}, {r['id'] for r in corpus()})
        self.assertEqual(len(workload_hash()), 64)
        self.assertEqual(len(performance_texts()), 16)
