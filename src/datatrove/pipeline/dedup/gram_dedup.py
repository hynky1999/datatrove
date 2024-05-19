import math
from typing import Any, TypeVar
from loguru import logger

from datatrove.data import DocumentsPipeline

from datatrove.pipeline.base import PipelineStep
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from datatrove.utils.text import TextNormConfig, simplify_text


REMOVE_SPANS_ELEMENT = TypeVar('REMOVE_SPANS_ELEMENT')
def remove_spans(elements: list[REMOVE_SPANS_ELEMENT], spans: list[tuple[int, int]]) -> list[REMOVE_SPANS_ELEMENT]:
    """
    Remove spans from elements
    Args:
        elements: List of elements
        spans: List of [start, end) tuples
    """
    elements_to_keep = []
    spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    cur_idx = 0
    for start, end in spans:
        start = max(cur_idx, start)
        elements_to_keep.extend(elements[cur_idx:start])
        cur_idx = end
    elements_to_keep.extend(elements[cur_idx:])
    return elements_to_keep
    
def select_non_overlapping_spans(spans_to_remove_list: list[tuple[int, int, int]]):
    """
    Selects spans which don't overlap, while tyring to maxime coverage of the spans
    Args:
        spans_to_remove_list: List of spans to remove
    """
    if len(spans_to_remove_list) == 0:
        return []

    # Sort the spans, first by k to cover with the biggest k-grams first, then by start
    # -> For same k start_i-1 <= start_i
    spans_to_remove_list = sorted(spans_to_remove_list, key=lambda x: (-x[2], x[0], -x[1]))
    last_start = spans_to_remove_list[0][2]
    cur_idx = 0
    # Rule: ordered by span[0] + no overlap
    # Use dummy to ensure that there is always a span to the left
    new_spans = [(-1,-1,1)]
    # Rule: new_spans[cur_idx][0] <= span[0]
    # Thus assuming for all span[1] > span[0] -> new_spans[cur_idx][1] > span[0]
    # The second assumption only doesn't hold for the dummy span at the start
    for span in spans_to_remove_list:
        new_span_start = span[0]
        new_span_end = span[1]
        # We expect that last_start >= span[0], otherwise we have to reset
        if span[0] < last_start:
            cur_idx = 0

        left_idx = cur_idx
        # Search for first left span, that is the last one for which s_start < span[0]
        while left_idx + 1 < len(new_spans) and new_spans[left_idx+1][0] < span[0]:
            left_idx += 1
        
        # If there is an overlap with the left span
        if left_idx != len(new_spans) and new_spans[left_idx][1] > span[0]:
            # Shorten left border to remove overlap
            shorten_times = math.ceil((new_spans[left_idx][1] - span[0]) / span[2])
            new_span_start = span[0] + span[2] * shorten_times

        # Then search for the first right span that is the first one for which s_end > span[1]
        right_idx = cur_idx
        while right_idx < len(new_spans) and new_spans[right_idx][1] <= span[1]:
            right_idx += 1
        
        # If there is an overlap with the right span
        if right_idx != len(new_spans) and new_spans[right_idx][0] < span[1]:
            # Shorten the right border to remove overlap
            shorten_times = math.ceil((span[1] - new_spans[right_idx][0]) / span[2])
            new_span_end = span[1] - span[2] * (shorten_times)

        # If the span is no longer valid don't add it and move the cur_idx
        if new_span_end - new_span_start < span[2] * 2:
            cur_idx = left_idx
            continue

        new_spans = new_spans[:left_idx+1] + [(new_span_start, new_span_end, span[2])] + new_spans[right_idx:]
        cur_idx = left_idx+1
            
    # Check that the spans are non-overlapping
    assert all(new_spans[i][0] < new_spans[i+1][0] for i in range(len(new_spans) - 1))
    # Remove the dummy span
    return new_spans[1:]


def get_dup_consequtive_kgram_ids(sequence: np.ndarray, tokens_cum_sum: np.ndarray, k: int, min_rep: int, min_size: int):
    """
    Takes a sequence and 2xI matrix of spans signifying the start and end of the span which are duplicated
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
    spans_with_min_size = tokens_cum_sum[spans[:, 1]-1] - tokens_cum_sum[np.max(spans[:, 0]-1, 0)] >= min_size
    spans = spans[spans_with_min_reps | spans_with_min_size]
    return spans

def get_dup_consequtive_mutli_kgram_ids(tokens: list[str], min_k: int, max_k: int, min_rep: int, min_size: int, max_split_size: int, split_overlap: int):
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
    spans_to_remove = []
    for start_split in range(0, len(tokens), max_split_size - split_overlap):
        end_split = start_split + max_split_size
        split_tokens = tokens[start_split:end_split]
        # Converting tokens to ints, make comparisson in next steps faster
        _, int_tokens = np.unique(split_tokens, return_inverse=True)
        tokens_cum_sum = np.cumsum([len(token) for token in split_tokens])
        for k in range(min_k, max_k + 1):
            new_spans_to_remove = get_dup_consequtive_kgram_ids(int_tokens, tokens_cum_sum, k, min_rep, min_size)
            offseted_idx_to_remove = [(start + start_split, end + start_split, k) for start, end in new_spans_to_remove]
            spans_to_remove.extend(offseted_idx_to_remove)
    
    filtered_spans_to_remove = select_non_overlapping_spans(spans_to_remove)
    with_removed_last = [(start, end-k) for start, end, k in filtered_spans_to_remove]
    return with_removed_last


        

        
    


def normalize_text(text):
    text = re.sub(r'\d+(\.\d+)?', '<NUMBER>', text)
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    for month in months:
        text = text.replace(month, '<MONTH>')
    return text


class GramDeduplication(PipelineStep):
    def __init__(self, text_norm_config: TextNormConfig = TextNormConfig(norm_numbers=False, remove_punctuation=False), deduplicated_md_tables: bool = False, deduplicate_over_lines: bool = False, min_span_size: int=45, min_k_gram: int = 1, max_k_gram: int = 10, min_rep: int = 1, span_size: int=20000):
        from nltk import NLTKWordTokenizer
        tokenizer = NLTKWordTokenizer()
        self.min_span_size = min_span_size
        self.min_k_gram = min_k_gram
        self.max_k_gram = max_k_gram
        self.min_rep = min_rep
        self.span_size = span_size
        self.deduplicated_md_tables = deduplicated_md_tables
        self.deduplicate_over_lines = deduplicate_over_lines
        self.text_norm_config = text_norm_config
        self.tokenizer = tokenizer
    
    def dedup_text(self, text: str):
        # Heuristic for not deduplicating tables
        if '|' in text and self.deduplicated_md_tables:
            return text

        spans = list(self.tokenizer.span_tokenize(text))
        tokens = [simplify_text(text[start:end], self.text_norm_config) for start, end in spans]
        # Since we are always removing preceeding repetitions, we want to also remove their whitespaces
        # Thus we use right delimiter
        tokens_with_right_delimiter = [text[span_prev[0]:span_next[0]] for span_prev, span_next in zip(spans[:-1], spans[1:])] + [text[spans[-1][0]:]]
        overlap = self.max_k_gram * (self.min_rep + 1)
        spans_to_remove = get_dup_consequtive_mutli_kgram_ids(tokens, self.min_k_gram, self.max_k_gram, self.min_rep, self.min_span_size, self.span_size, overlap)
        if len(spans_to_remove) == 0:
            return text
        
        tokens_with_right_delimiter = remove_spans(tokens_with_right_delimiter, spans_to_remove)

        new_line = ''.join(tokens_with_right_delimiter)
        return new_line

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            # Prefer line by line 
            lines = document.text.split("\n") if self.deduplicate_over_lines else [document.text]
            deduplicated_lines = [self.dedup_text(line) for line in lines]
            document.text = "\n".join(deduplicated_lines)
        return data



class LineDeduplication(PipelineStep):
    def __init__(self, text_norm_config: TextNormConfig = TextNormConfig(norm_numbers=False, remove_punctuation=False), min_span_size: int=1, min_k_gram: int = 1, max_k_gram: int = 3, min_rep: int = 1, span_size: int=20000):
        if min_span_size < 1:
            logger.warning("It's recommended to have min_span_size >= 1 so that consecutive newlines are not considered duplicates")

        super().__init__()

        from nltk import NLTKWordTokenizer
        tokenizer = NLTKWordTokenizer()
        self.min_span_size = min_span_size
        self.min_k_gram = min_k_gram
        self.max_k_gram = max_k_gram
        self.min_rep = min_rep
        self.text_norm_config = text_norm_config
        self.span_size = span_size
        self.tokenizer = tokenizer
    

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            # Prefer line by line 
            lines = document.text.split("\n")
            overlap = self.max_k_gram * (self.min_rep + 1)
            simplified_lines = [simplify_text(line, self.text_norm_config) for line in lines]
            span_to_remove = get_dup_consequtive_mutli_kgram_ids(
                simplified_lines,
                self.min_k_gram, self.max_k_gram, self.min_rep, self.min_span_size, self.span_size, overlap)
            if len(span_to_remove) > 0:
                # Use non-simplified lines
                document.text = "\n".join(remove_spans(lines, span_to_remove))
        return data

