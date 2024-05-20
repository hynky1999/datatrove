import bisect
from dataclasses import dataclass, field
import math
from typing import Any, Literal, TypeVar
from loguru import logger

from datatrove.data import DocumentsPipeline

from datatrove.pipeline.base import PipelineStep
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from datatrove.utils.text import TextNormConfig, simplify_text

CONDITION_MODES = Literal["&", "|"]

@dataclass(frozen=True)
class KgramDeduplicationConfig:
    """
    Configuration for in-document kgram deduplication
    deduplicate_over_lines: Whether to deduplicate over lines or tokens
    deduplicated_md_tables: Whether to deduplicate markdown tables
    text_norm_config: Text normalization configuration, before kgram deduplication
    min_rep_size: Minimum size (chars) ofk-gram duplicates to consider for removal
    min_reps: Minimum number of consecutive k-gram duplicates to consider for removal
    min_k_gram: Min k-gram for removal
    max_k_gram: Max k-gram for removal
    condition_mode: Condition mode when evaluating min_rep_size/min_reps
    chunk_size: Sizes of chunks to conider, to chunk the deduplicated lines/tokens into before performing the deduplication.
        This is to avoid memory overflow.
    """
    deduplicate_over_lines: bool = True
    deduplicated_md_tables: bool = True
    text_norm_config: TextNormConfig = field(default_factory=lambda: TextNormConfig(norm_numbers=False, remove_punctuation=False, norm_whitespace=False, norm_unicode_diacritics=False, lowercase=True))
    min_rep_size: int = 70
    min_reps: int = 25
    min_k_gram: int = 1
    max_k_gram: int = 10
    condition_mode: CONDITION_MODES = "|"
    chunk_size: int = 100000

DEFAULT_GRAM_DEDUP_CONFIG = KgramDeduplicationConfig()

@dataclass(frozen=True)
class LineDeduplicationConfig:
    """
    Configuration for in-document kgram deduplication
    text_norm_config: Text normalization configuration, before kgram deduplication
    min_rep_size: Minimum size (chars) ofk-gram duplicates to consider for removal
    min_reps: Minimum number of consecutive k-gram duplicates to consider for removal
    min_k_gram: Min k-gram for removal
    max_k_gram: Max k-gram for removal
    condition_mode: Condition mode when evaluating min_rep_size/min_reps
    chunk_size: Sizes of chunks to conider, to chunk the deduplicated lines/tokens into before performing the deduplication.
        This is to avoid memory overflow.
    """
    text_norm_config: TextNormConfig = field(default_factory=lambda: TextNormConfig(norm_numbers=False, remove_punctuation=False, norm_whitespace=False, norm_unicode_diacritics=False, lowercase=True))
    min_rep_size: int = 1
    min_reps: int = 1
    min_k_gram: int = 1
    max_k_gram: int = 3
    condition_mode: CONDITION_MODES = "&"
    chunk_size: int = 20000

DEFAULT_LINE_DEDUP_CONFIG = LineDeduplicationConfig()

@dataclass
class Span:
    """
    Span of a elements to remove
    start: Start of the span (inclusive)
    end: End of the span (exclusive)
    k: k-gram size
    """
    start: int
    end: int
    k: int



REMOVE_SPANS_ELEMENT = TypeVar('REMOVE_SPANS_ELEMENT')
def remove_spans(elements: list[REMOVE_SPANS_ELEMENT], spans: list[Span]) -> list[REMOVE_SPANS_ELEMENT]:
    """
    Remove spans of elements
    Args:
        elements: List of elements
        spans: List of spans (Expected to be non-overlapping, otherwise the result is undefined)
    Returns:
        List of elements without the spans
    """
    elements_to_keep = []
    spans = sorted(spans, key=lambda x: (x.start, -x.end))
    cur_idx = 0
    for span in spans:
        start = max(cur_idx, span.start)
        elements_to_keep.extend(elements[cur_idx:start])
        cur_idx = span.end
    elements_to_keep.extend(elements[cur_idx:])
    return elements_to_keep
    
def select_non_overlapping_spans(spans_to_remove_list: list[Span]) -> list[Span]:
    """
    Selects spans which don't overlap, while tyring to maxime coverage of the spans.
        If the span doesn't fit, it tries to shorten the span to fit (always by span.k).
    Args:
        spans_to_remove_list: List of spans, where span.start <= span.end
    
    Returns:
        List of non-overlapping spans, sorted by span.start
    """
    if len(spans_to_remove_list) == 0:
        return []

    # Sort the spans, first by span_size to cover with the biggest reps first, then by start
    # -> For same k start_i-1 <= start_i
    spans_to_remove_list = sorted(spans_to_remove_list, key=lambda x: (-(x.end-x.start), x.start, -x.end))
    # Use dummy to ensure that there is always a span to the left
    selected_spans: list[Span] = [Span(-1,-1,1)]
    for span in spans_to_remove_list:
        new_span_end, new_span_start = span.end, span.start

        # Index of the span which is the last one for which selected_spans[left_idx].start < span.start (closest to the left)
        left_idx = bisect.bisect_left(selected_spans, span.start, key=lambda x: x.start) - 1
        
        # If there is an overlap with the left span
        if left_idx != len(selected_spans) and selected_spans[left_idx].end > span.start:
            # Shorten left border to remove overlap
            shorten_times = math.ceil((selected_spans[left_idx].end - span.start) / span.k)
            new_span_start = span.start + span.k * shorten_times

        # Index of the span which is the first one for which selected_spans[left_idx].end < span.end
        right_idx = bisect.bisect_right(selected_spans, span.end, key=lambda x: x.end, lo=left_idx)
        
        # If there is an overlap with the right span
        if right_idx != len(selected_spans) and selected_spans[right_idx].start < span.end:
            # Shorten the right border to remove overlap
            shorten_times = math.ceil((span.end - selected_spans[right_idx].start) / span.k)
            new_span_end = span.end - span.k * (shorten_times)

        # If the span is no longer valid don't add it
        if new_span_end - new_span_start < span.k * 2:
            continue

        selected_spans = selected_spans[:left_idx+1] + [Span(new_span_start, new_span_end, span.k)] + selected_spans[right_idx:]
            
    # Check that the spans are non-overlapping
    assert all(selected_spans[i].start < selected_spans[i+1].start for i in range(len(selected_spans) - 1))
    # Remove the dummy span
    return selected_spans[1:]


def get_dup_consequtive_kgram_ids(sequence: np.ndarray, tokens_cum_sum: np.ndarray, k: int, min_rep: int, min_size: int, condition_mode: str):
    """
    Takes a sequence and returns 2xI matrix of spans signifying the start and end of the span which are duplicated
    """
    if len(sequence) // 2 < k:
        return np.array([])
    view = sliding_window_view(sequence, k)

    # Do array wise comparisson
    same_sequences = np.all(view[:-k] == view[k:], axis=1)

    def start_stop(a):
        # "Enclose" mask with sentients to catch shifts later on
        mask = np.r_[False,a,False]

        # Get the shifting indices
        idx = np.flatnonzero(mask[1:] != mask[:-1])

        # Get the start and end indices with slicing along the shifting ones
        return np.column_stack((idx[::2], idx[1::2]))

    spans = []
    for shift in range(k):
        idx = start_stop(same_sequences[shift::k])
        real_idx = idx * k + shift
        spans.extend(real_idx)

    # sort by first and then second column
    if len(spans) == 0:
        return np.array([])
    spans = np.vstack(spans)
    spans[:, 1] += k
    spans = spans[np.argsort(spans[:, 0])]

    spans_with_min_reps = (spans[:, 1] - spans[:, 0]) // k >= min_rep + 1
    # The end is non-inclusive so we have to use -1
    # To get size between start and end from cumsums we have access start-1
    spans_with_min_size = tokens_cum_sum[spans[:, 1]-1] - tokens_cum_sum[np.maximum(spans[:, 0]-1, 0)] >= min_size
    spans_condition = spans_with_min_reps
    if condition_mode == "&":
        spans_condition = spans_condition & spans_with_min_size
    elif condition_mode == "|":
        spans_condition = spans_condition | spans_with_min_size
    spans = spans[spans_condition]
    return spans

def get_dup_consequtive_mutli_kgram_ids(tokens: list[str], min_k: int, max_k: int, min_rep: int, min_size: int, condition_mode: str, max_split_size: int, split_overlap: int):
    """
    Takes list of strings and returns [start, end) spans of duplicated consecutive k-grams
    When choosing which to remove the first k-grams are always chosen.

    Example:
    k = 2
    tokens = ["a", "b", "a", "b", "d"]
    -> [0, 1]

    Args:
        tokens: List of tokens
        min_k: Minimum k-gram size
        max_k: Maximum k-gram size
        min_rep: Minimum repetition of k-gram
        min_size: Minimum size of k-gram
        max_split: Maximum number of tokens considered in each split. This is to avoid memory overflow.
        split_overlap: Number of tokens to overlap between splits. This is to avoid duplicates being removed in the same split.
    """


    # First convert string tokens to integers, for multiple k this allows faster comparisson
    spans_to_remove: list[Span] = []
    for start_split in range(0, len(tokens), max_split_size - split_overlap):
        end_split = start_split + max_split_size
        split_tokens = tokens[start_split:end_split]
        # Converting tokens to ints, make comparisson in next steps faster
        _, int_tokens = np.unique(split_tokens, return_inverse=True)
        tokens_cum_sum = np.cumsum([len(token) for token in split_tokens])
        for k in range(min_k, max_k + 1):
            new_spans_to_remove = get_dup_consequtive_kgram_ids(int_tokens, tokens_cum_sum, k, min_rep, min_size, condition_mode)
            new_span_with_k = [Span(start + start_split, end + start_split, k) for start, end in new_spans_to_remove]
            spans_to_remove.extend(new_span_with_k)
    
    filtered_spans_to_remove = select_non_overlapping_spans(spans_to_remove)
    with_removed_last = [Span(span.start, span.end-span.k, span.k) for span in filtered_spans_to_remove]
    return with_removed_last


class GramDeduplication(PipelineStep):
    def __init__(self, dedup_config: KgramDeduplicationConfig = DEFAULT_GRAM_DEDUP_CONFIG):
        from nltk import NLTKWordTokenizer
        tokenizer = NLTKWordTokenizer()
        self.tokenizer = tokenizer
        self.dedup_config = dedup_config
    
    def dedup_text(self, text: str):
        # Heuristic for not deduplicating tables
        if '|' in text and not self.dedup_config.deduplicated_md_tables:
            return text

        spans = list(self.tokenizer.span_tokenize(text))
        tokens = [simplify_text(text[start:end], self.dedup_config.text_norm_config) for start, end in spans]
        # Since we are always removing preceeding repetitions, we want to also remove their whitespaces
        # Thus we use right delimiter
        tokens_with_right_delimiter = [text[span_prev[0]:span_next[0]] for span_prev, span_next in zip(spans[:-1], spans[1:])] + [text[spans[-1][0]:]]
        overlap = self.dedup_config.max_k_gram * (self.dedup_config.min_reps + 1)
        spans_to_remove = get_dup_consequtive_mutli_kgram_ids(tokens, self.dedup_config.min_k_gram, self.dedup_config.max_k_gram, self.dedup_config.min_reps, self.dedup_config.min_rep_size, self.dedup_config.condition_mode, self.dedup_config.chunk_size, overlap)
        if len(spans_to_remove) == 0:
            return text
        
        tokens_with_right_delimiter = remove_spans(tokens_with_right_delimiter, spans_to_remove)

        new_line = ''.join(tokens_with_right_delimiter)
        return new_line

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            # Prefer line by line 
            lines = document.text.split("\n") if not self.dedup_config.deduplicate_over_lines else [document.text]
            deduplicated_lines = [self.dedup_text(line) for line in lines]
            document.text = "\n".join(deduplicated_lines)
        return data



class LineDeduplication(PipelineStep):
    def __init__(self, dedup_config: LineDeduplicationConfig = DEFAULT_LINE_DEDUP_CONFIG):
        self.dedup_config = dedup_config
        if self.dedup_config.min_rep_size < 1:
            logger.warning("It's recommended to have min_rep_size >= 1 so that consecutive newlines are not considered duplicates")

        super().__init__()

        from nltk import NLTKWordTokenizer
        tokenizer = NLTKWordTokenizer()
        self.tokenizer = tokenizer
    

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            # Prefer line by line 
            lines = document.text.split("\n")
            overlap = self.dedup_config.max_k_gram * (self.dedup_config.min_reps + 1)
            simplified_lines = [simplify_text(line, self.dedup_config.text_norm_config) for line in lines]
            span_to_remove = get_dup_consequtive_mutli_kgram_ids(
                simplified_lines,
                self.dedup_config.min_k_gram, self.dedup_config.max_k_gram, self.dedup_config.min_reps, self.dedup_config.min_rep_size, self.dedup_config.condition_mode, self.dedup_config.chunk_size, overlap)
            if len(span_to_remove) > 0:
                # Use non-simplified lines
                document.text = "\n".join(remove_spans(lines, span_to_remove))
        return data

