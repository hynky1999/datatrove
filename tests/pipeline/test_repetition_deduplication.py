import unittest
from datatrove.pipeline.dedup.gram_dedup import GramDeduplication, LineDeduplication
from datatrove.pipeline.base import Document


class TestRepetitionDeduplication(unittest.TestCase):
    def test_gram_deduplication(self):
        docs = [Document("e? Can you? Can you?” Well", "0")]
        gram_dedup = GramDeduplication(min_span_size = 0)
        dedup_docs = gram_dedup.run(docs)
        expected = "e? Can you?” Well"
        self.assertEqual(dedup_docs[0].text, expected)
    
    def test_line_deduplication(self):
        docs = [Document("A\n\nB\n\nB\nC\n\nD", "0")]
        line_dedup = LineDeduplication()
        dedup_docs = line_dedup.run(docs)
        expected = "A\n\nB\nC\n\nD"
        self.assertEqual(dedup_docs[0].text, expected)

