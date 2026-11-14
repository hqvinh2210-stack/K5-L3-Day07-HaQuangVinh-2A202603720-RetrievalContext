import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from retrieval import HybridRetriever, InterleaveHybridRetriever, TfidfRetriever, metadata_matches


def chunk(chunk_id, text, section="chuong_i"):
    return {
        "id": chunk_id,
        "source_id": "doc",
        "source": "doc.pdf",
        "page": 1,
        "text": text,
        "metadata": {"section": section},
    }


class FakeRetriever:
    def __init__(self, rows, strategy):
        self.rows = [{**row, "strategy": strategy, "score": 1.0} for row in rows]

    def search(self, query, *, filters, top_k):
        return self.rows[:top_k]


class RetrievalTests(unittest.TestCase):
    def test_metadata_filter_and_tfidf_contract(self):
        rows = [chunk("a", "hoàn tiền bảy ngày"), chunk("b", "giao hàng", "chuong_ii")]
        self.assertTrue(metadata_matches(rows[0], {"section": "chuong_i"}))
        self.assertFalse(metadata_matches(rows[0], {"section": "chuong_ii"}))
        result = TfidfRetriever(rows).search(
            "hoàn tiền", filters={"section": "chuong_i"}, top_k=3
        )
        self.assertEqual([row["id"] for row in result], ["a"])
        self.assertEqual(result[0]["strategy"], "tfidf")

    def test_hybrid_rrf_fuses_and_deduplicates(self):
        a, b, c = chunk("a", "a"), chunk("b", "b"), chunk("c", "c")
        dense = FakeRetriever([a, b], "dense")
        lexical = FakeRetriever([b, c], "tfidf")
        result = HybridRetriever(dense, lexical, candidate_k=3).search(
            "q", filters=None, top_k=3
        )
        self.assertEqual([row["id"] for row in result], ["b", "a", "c"])
        self.assertEqual(len({row["id"] for row in result}), 3)
        self.assertEqual(result[0]["component_ranks"], {"dense": 2, "tfidf": 1})

    def test_hybrid_weights_change_ranking(self):
        a, b = chunk("a", "a"), chunk("b", "b")
        dense = FakeRetriever([a, b], "dense")
        lexical = FakeRetriever([b, a], "tfidf")
        result = HybridRetriever(
            dense, lexical, candidate_k=2, dense_weight=2.0
        ).search("q", filters=None, top_k=2)
        self.assertEqual([row["id"] for row in result], ["a", "b"])

    def test_interleave_deduplicates_and_alternates(self):
        a, b, c = chunk("a", "a"), chunk("b", "b"), chunk("c", "c")
        dense = FakeRetriever([a, b], "dense")
        lexical = FakeRetriever([a, c], "tfidf")
        result = InterleaveHybridRetriever(dense, lexical, candidate_k=2).search(
            "q", filters=None, top_k=3
        )
        self.assertEqual([row["id"] for row in result], ["a", "b", "c"])
        self.assertEqual([row["selected_from"] for row in result], ["dense", "dense", "tfidf"])


if __name__ == "__main__":
    unittest.main()
