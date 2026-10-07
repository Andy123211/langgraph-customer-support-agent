"""Deterministic unit tests for the lexical retrieval and RRF components."""

import unittest

from src.support_agent.retrieval import (
    bm25_relevance,
    bm25_scores,
    reciprocal_rank_fusion,
    tokenize,
)


class TestHybridRetrievalHelpers(unittest.TestCase):
    def test_bm25_ranks_exact_keyword_match_first(self):
        documents = {
            "shipping": "Shipping delivery options and tracking information",
            "returns": "Return policy and refund processing instructions",
        }

        scores = bm25_scores("refund return", documents)

        self.assertGreater(scores["returns"], scores["shipping"])
        self.assertEqual(scores["shipping"], 0)

    def test_bm25_tokenizes_chinese_phrases_into_bigrams(self):
        self.assertEqual(tokenize("退货政策"), ["退货", "货政", "政策"])
        self.assertGreater(bm25_scores("退货", {"returns": "退货政策"})["returns"], 0)

    def test_rrf_promotes_documents_ranked_by_both_retrievers(self):
        fused = reciprocal_rank_fusion(
            ["keyword-hit", "keyword-only"],
            ["keyword-hit", "vector-only"],
        )

        self.assertEqual(fused[0], "keyword-hit")
        self.assertEqual(set(fused), {"keyword-hit", "keyword-only", "vector-only"})

    def test_bm25_relevance_is_bounded_and_zero_safe(self):
        self.assertEqual(bm25_relevance(0), 0)
        self.assertGreater(bm25_relevance(1), 0)
        self.assertLess(bm25_relevance(1), 1)


if __name__ == "__main__":
    unittest.main()
