import unittest
from datatrove.pipeline.dedup.gram_dedup import DEFAULT_GRAM_DEDUP_CONFIG, DEFAULT_LINE_DEDUP_CONFIG, GramDeduplication, LineDeduplication, Span, select_non_overlapping_spans
from datatrove.pipeline.base import Document
from dataclasses import replace

def span_overlapping(spans: list[Span]) -> bool:
    last_end = 0
    for span in spans:
        if span.start < last_end:
            return True
        last_end = span.end
    return False

def span_coverage(spans: list[Span]) -> int:
    return sum([span.end - span.start for span in spans])


class TestRepetitionDeduplication(unittest.TestCase):
    def test_gram_deduplication(self):
        docs = [Document("e? Can you? Can you?” Well", "0")]
        gram_dedup = GramDeduplication(dedup_config=replace(DEFAULT_GRAM_DEDUP_CONFIG, min_rep_size=0))
        dedup_docs = gram_dedup.run(docs)
        expected = "e? Can you?” Well"
        self.assertEqual(dedup_docs[0].text, expected)
    
    def test_line_deduplication(self):
        docs = [Document("A\n\nB\n\nB\nC\n\nD", "0")]
        line_dedup = LineDeduplication(dedup_config=replace(DEFAULT_LINE_DEDUP_CONFIG, min_rep_size=0))
        dedup_docs = line_dedup.run(docs)
        expected = "A\n\nB\nC\n\nD"
        self.assertEqual(dedup_docs[0].text, expected)
    
    def test_span_selection(self):
        spans_to_remove_list = [
            Span(0, 4, 2), Span(4, 6, 1), Span(0, 1, 1),
        ]

        selected_spans = select_non_overlapping_spans(spans_to_remove_list)
        # self.assertEqual(selected_spans, [(0, 1, 1)])
        self.assertEqual(span_coverage(selected_spans), 6)

    def test_span_selection_non_overlapping_1(self):
        spans_to_remove_list = [
            Span(1, 5, 2), Span(5, 7, 1), Span(0, 10, 1)
        ]

        selected_spans = select_non_overlapping_spans(spans_to_remove_list)
        self.assertEqual(span_overlapping(selected_spans), False)
        self.assertEqual(span_coverage(selected_spans), 10)

    def test_span_selection_non_overlapping_2(self):
        spans_to_remove_list = [
            Span(1, 5, 2), Span(20, 22, 1)
        ]

        selected_spans = select_non_overlapping_spans(spans_to_remove_list)
        self.assertEqual(span_overlapping(selected_spans), False)
        self.assertEqual(span_coverage(selected_spans), 6)






