import re
from loguru import logger

from datatrove.data import DocumentsPipeline

from datatrove.pipeline.base import PipelineStep
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

    

def get_dup_consequtive_kgram_ids(sequence: np.ndarray, tokens_cum_sum: np.ndarray, k: int, min_rep: int, min_size: int):
    """
    Takes a sequence and returns a list of indexes to remove
    """
    if len(sequence) // 2 < k:
        return []
    view = sliding_window_view(sequence, k)

    # Do array wise comparisson
    same_sequences = np.all(view[:-k] == view[k:], axis=1)

    indx_to_remove = []

    def start_stop(a):
        # "Enclose" mask with sentients to catch shifts later on
        mask = np.r_[False,a,False]

        # Get the shifting indices
        idx = np.flatnonzero(mask[1:] != mask[:-1])

        # Get the start and end indices with slicing along the shifting ones
        return np.column_stack((idx[::2], idx[1::2] - 1))

    # Guilherme is genius
    def process_sequence(cum_sum, same_sequences, k, min_size, min_rep):
        indexes = []
        for shift in range(k):
            idx = start_stop(same_sequences[shift::k])
            real_idx = idx * k + shift
            indexes.extend(real_idx)

        # sort by first and then second column
        if len(indexes) == 0:
            return []
        indexes = np.vstack(indexes)
        indexes = indexes[np.argsort(indexes[:, 0])]

        selected_indexes_reps_where = (indexes[:, 1] - indexes[:, 0]) // k >= min_rep
        selected_cumulative_sums = cum_sum[indexes]
        selected_indexes_size = selected_cumulative_sums[:, 1] - selected_cumulative_sums[:, 0] > min_size
        indexes = indexes[selected_indexes_size | selected_indexes_reps_where]
        # Convert -1,2 matrix to list of tuples
        spans = indexes.tolist()
        return spans

    indx_to_remove = process_sequence(tokens_cum_sum, same_sequences, k, min_size, min_rep)
    
    return indx_to_remove

def get_dup_consequtive_mutli_kgram_ids(tokens: list[str], min_k: int, max_k: int, min_rep: int, min_size: int):
    """
    Takes tokens and returns indexes of token, which are duplicated consequtive k-gram.
    When choosing which to remove the first is always chosen.

    To allow 

    Example:
    k = 2
    tokens = ["a", "b", "a", "b", "d"]
    -> [0, 1]
    """
    # First convert string tokens to integers, for multiple k this allows faster comparisson
    _, int_tokens = np.unique(tokens, return_inverse=True)
    tokens_cum_sum = np.array([len(token) for token in tokens])
    all_idx_to_remove = set()
    for k in range(max_k, min_k - 1, -1):
        idx_to_remove = get_dup_consequtive_kgram_ids(int_tokens, tokens_cum_sum, k, min_rep, min_size)
        all_idx_to_remove.update(set(tuple(x) for x in idx_to_remove))
    
    return list(all_idx_to_remove)


class GramDeduplication(PipelineStep):
    def __init__(self, min_span_size: int=45, min_k_gram: int = 1, max_k_gram: int = 10, min_rep: int = 30):
        from nltk import NLTKWordTokenizer
        tokenizer = NLTKWordTokenizer()
        self.min_span_size = min_span_size
        self.min_k_gram = min_k_gram
        self.max_k_gram = max_k_gram
        self.min_rep = min_rep
        self.tokenizer = tokenizer

    def dedup_line(self, line: str):
        # Heuristic for not deduplicating tables
        if '|' in line:
            return line

        spans = list(self.tokenizer.span_tokenize(line))
        tokens = [line[start:end] for start, end in spans]
        # Since we are always removing preceeding repetitions, we want to also remove their whitespaces
        # Thus we use right delimiter
        tokens_with_right_delimiter = [line[span_prev[0]:span_next[0]] for span_prev, span_next in zip(spans[:-1], spans[1:])] + [line[spans[-1][0]:]]
        idx_to_remove = get_dup_consequtive_mutli_kgram_ids(tokens, self.min_k_gram, self.max_k_gram, self.min_rep, self.min_span_size)
        if len(idx_to_remove) == 0:
            return line
        
        tokens_with_right_delimiter = np.delete(tokens_with_right_delimiter, list(idx_to_remove))

        new_line = ''.join(tokens_with_right_delimiter)
        return new_line

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            # Prefer line by line 
            lines = document.text.split("\n")
            deduplicated_lines = [self.dedup_line(line) for line in lines]
            document.text = "\n".join(deduplicated_lines)
        return data


class LineDeduplication(PipelineStep):
    def __init__(self, min_span_size: int=1, min_k_gram: int = 1, max_k_gram: int = 3, min_rep: int = 30):
        if min_span_size < 1:
            logger.warning("It's recommended to have min_span_size >= 1 so that consecutive newlines are not considered duplicates")

        from nltk import NLTKWordTokenizer
        tokenizer = NLTKWordTokenizer()
        self.min_span_size = min_span_size
        self.min_k_gram = min_k_gram
        self.max_k_gram = max_k_gram
        self.min_rep = min_rep
        self.tokenizer = tokenizer

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            # Prefer line by line 
            lines = document.text.split("\n")
            idx_to_remove = get_dup_consequtive_mutli_kgram_ids(lines, self.min_k_gram, self.max_k_gram, self.min_rep, self.min_span_size)
            if len(idx_to_remove) > 0:
                new_lines = np.delete(lines, idx_to_remove)
                document.text = "\n".join(new_lines)
        return data

