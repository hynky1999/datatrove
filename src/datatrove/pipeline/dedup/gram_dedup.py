import re
from loguru import logger

from datatrove.data import DocumentsPipeline

from datatrove.pipeline.base import PipelineStep
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def get_dup_consequtive_mutli_kgram_ids(tokens: list[str], min_k: int, max_k: int, min_rep: int, min_size: int):
    """
    Takes tokens and returns indexes of token, which are duplicated consequtive k-gram.
    When choosing which to remove the preceeding is always chosen.

    To allow 

    Example:
    k = 2
    tokens = ["a", "b", "a", "b", "d"]
    -> [0, 1]
    """
    tokens_arr = np.array(tokens)
    all_idx_to_remove = set()
    for k in range(max_k, min_k - 1, -1):
        idx_to_remove = get_dup_consequtive_kgram_ids(tokens_arr, k, min_rep, min_size)
        all_idx_to_remove.update(idx_to_remove)
    
    return list(all_idx_to_remove)
    

def get_dup_consequtive_kgram_ids(sequence: np.ndarray, k: int, min_rep: int, min_size: int):
    """
    Takes a sequence and returns a list of indexes to remove
    """
    if len(sequence) // 2 < k:
        return []
    view = sliding_window_view(sequence[:-k], k)
    view_plus_k = sliding_window_view(sequence[k:], k)

    # Do array wise comparisson
    same_sequences = np.all(view == view_plus_k, axis=1)
    i = 0
    indx_to_remove = []

    # Guilherme is genius
    cumulative_sum = np.cumsum([0] + [len(token) for token in sequence])
    while i < len(same_sequences):
        new_possible_start = i
        # Jump over the duplicates till you find a non-duplicate
        while new_possible_start < len(same_sequences) and same_sequences[new_possible_start]:
            new_possible_start += k

        # If we didn't jump at all
        if new_possible_start == i:
            i += 1
        else:
            repetition_length = cumulative_sum[new_possible_start + k] - cumulative_sum[i]
            reps = (new_possible_start - i) // k
            # If the repetition satisfies constraints add indxes to remove
            if repetition_length >= min_size or reps >= min_rep:
                indx_to_remove.extend(range(i, new_possible_start))
            # In both cases we move the index, in case on not satisfaction
            # this speed-up the search
            i = new_possible_start
    
    return indx_to_remove

def normalize_text(text):
    text = re.sub(r'\d+(\.\d+)?', '<NUMBER>', text)
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    for month in months:
        text = text.replace(month, '<MONTH>')
    return text


class GramDeduplication(PipelineStep):
    name = "Gram Deduplication"
    _requires_dependencies = ["nltk"]

    def __init__(self, min_span_size: int=45, min_k_gram: int = 1, max_k_gram: int = 10, min_rep: int = 30):
        super().__init__()
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
        if len(spans) <= 0:
            return line
        tokens = [line[start:end] for start, end in spans]
        # Since we are always removing preceeding repetitions, we want to also remove their whitespaces
        # Thus we use right delimiter
        idx_to_remove = get_dup_consequtive_mutli_kgram_ids(tokens, self.min_k_gram, self.max_k_gram, self.min_rep, self.min_span_size)
        if len(idx_to_remove) == 0:
            return line
        tokens_with_right_delimiter = [line[span_prev[0]:span_next[0]] for span_prev, span_next in zip(spans[:-1], spans[1:])] + [line[spans[-1][0]:]]
        
        tokens_with_right_delimiter = np.delete(tokens_with_right_delimiter, list(idx_to_remove))

        new_line = ''.join(tokens_with_right_delimiter)
        return new_line

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.track_time():
            for document in data:
                # Prefer line by line 
                lines = document.text.split("\n")
                deduplicated_lines = [self.dedup_line(line) for line in lines]
                document.text = "\n".join(deduplicated_lines)
                yield document


class LineDeduplication(PipelineStep):
    name = "Line Deduplication"
    _requires_dependencies = ["nltk"]

    def __init__(self, min_span_size: int=1, min_k_gram: int = 1, max_k_gram: int = 3, min_rep: int = 30, lowercase: bool = False, normalize: bool = False):
        if min_span_size < 1:
            logger.warning("It's recommended to have min_span_size >= 1 so that consecutive newlines are not considered duplicates")

        super().__init__()

        from nltk import NLTKWordTokenizer
        tokenizer = NLTKWordTokenizer()
        self.min_span_size = min_span_size
        self.min_k_gram = min_k_gram
        self.max_k_gram = max_k_gram
        self.min_rep = min_rep
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.normalize = normalize

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.track_time():
            for document in data:
                # Prefer line by line 
                lines = document.text.split("\n")
                norm_lines = lines
                if self.normalize:
                    norm_lines = [normalize_text(line) for line in lines]
                if self.lowercase:
                    norm_lines = [line.lower() for line in norm_lines]
                

                idx_to_remove = get_dup_consequtive_mutli_kgram_ids(norm_lines, self.min_k_gram, self.max_k_gram, self.min_rep, self.min_span_size)
                if len(idx_to_remove) > 0:
                    # Do this on original lines not normalized
                    new_lines = np.delete(lines, idx_to_remove)
                    document.text = "\n".join(new_lines)
                    logger.info(f"Removed {len(idx_to_remove)} lines from document {document.id}")
                
                yield document

