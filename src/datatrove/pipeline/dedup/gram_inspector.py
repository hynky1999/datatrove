from collections import defaultdict
from functools import partial
import hashlib
import heapq
from dataclasses import dataclass, replace
import math
from typing import Callable, Generator, Iterable

import numpy as np
from loguru import logger
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolder, DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from .utils import ExtensionHelperTN, simplify_text
import mmh3

DTYPE = np.uint32 | np.uint64


def mmh3_hash64(text: str, seed: int) -> int:
    return mmh3.hash64(text, seed=seed)[0]


@dataclass
class BloomCounterConfig:
    size: int
    n_hash_fcs: int
    dtype: DTYPE
    seed: int = 0

    @classmethod
    def get_optimal_k(cls, arr_size: int, expected_elements: int) -> int:
        return math.ceil(arr_size / expected_elements * np.log(2))

    @classmethod
    def from_expected_elements(
        cls,
        expected_elements: int,
        expected_mem_gb: int,
        dtype: DTYPE,
        seed: int = 0,
    ):
        """
        Given expected number of unique elements and expected memory consumption in GB,
        returns optimal config given such constraints
        """
        size_in_bytes = expected_mem_gb * 2**30
        arr_size = int(size_in_bytes / (np.dtype(dtype).itemsize))
        return cls(
            n_hash_fcs=cls.get_optimal_k(arr_size, expected_elements),
            dtype=dtype,
            size=arr_size,
            seed=seed,
        )

    def prob_incorrect_count(self, expected_elements: int) -> float:
        """
        Returns the probability of incorrect count estimation
        """
        return (
            1 - np.exp(-self.n_hash_fcs * expected_elements / self.size)
        ) ** self.n_hash_fcs


@dataclass
class NgramsConfig:
    n: int = 3
    char_level: bool = False


DEFAULT_GRAM_FINDER_CONFIG = NgramsConfig()


def get_ngrams(doc: Document, config: NgramsConfig) -> Generator[str, None, None]:
    from nltk import ngrams, word_tokenize

    text = simplify_text(doc.text, remove_punctuation=False)
    tokens = text if config.char_level else word_tokenize(text)
    join_char = " " if not config.char_level else ""
    n_grams = (join_char.join(ngram) for ngram in ngrams(tokens, config.n))
    return n_grams


class BloomCounter:
    """
    Implementation of a bloom filter counter.
    Probability of collisions = (1 - e^(-n/m))^k
    https://en.wikipedia.org/wiki/Bloom_filter

    n = number of inserted elements
    m = size of the filter in bits
    k = number of hash functions (optimal = m/n * ln(2))
    """

    def __init__(self, config: BloomCounterConfig):
        if config.n_hash_fcs > 100:
            logger.warning(
                f"Bloom filter has {config.n_hash_fcs} hash functions, which will slow down the processing, consider using less hash functions"
            )

        assert config.size < 2**64, "Bloom filter size is too big"

        self.seed = config.seed
        self.k = config.n_hash_fcs
        # This should hopefuly give us k independent hash functions, we can't really use
        # universal hashing, as we need over 32 bits :/
        self.hashers = [partial(mmh3_hash64, seed=self.seed + i) for i in range(self.k)]
        self._bloom_array = np.zeros(
            config.size,
            dtype=config.dtype,
        )

        self._parameters = None

    def add(self, ngrams: Iterable[str]):
        """
        Adds an gram to the bloom filter
        """
        indexes = self.get_indexes(self._get_shingles(ngrams))
        if indexes.size == 0:
            return
        # Sligthly faster than np.add.at
        idx, cnt = np.unique(indexes.reshape(-1), return_counts=True)
        new_cnt = cnt.astype(self._bloom_array.dtype) + self._bloom_array[idx]
        overflow_mask = new_cnt < self._bloom_array[idx]  # Check for overflow

        self._bloom_array[idx] = np.where(
            overflow_mask, np.iinfo(self._bloom_array.dtype).max, new_cnt
        )

    def get(self, ngrams: Iterable[str]) -> list[int]:
        indexes = self.get_indexes(self._get_shingles(ngrams))
        # The values stored in bloom filter are upper bound for the actual count
        if not indexes.size:
            return []
        return np.min(self._bloom_array[indexes], axis=1).tolist()

    def _get_shingles(self, ngrams: Iterable[str]) -> np.ndarray:
        """
        Returns the count of the ngram in the bloom filter
        """
        x = list(ngrams)
        return np.fromiter(
            (hasher(ngram) for ngram in x for hasher in self.hashers),
            dtype=self._bloom_array.dtype,
        ).reshape(-1, self.k)

    @classmethod
    def from_array(
        cls, bloom_array: np.ndarray, config: BloomCounterConfig
    ) -> "BloomCounter":
        assert bloom_array.shape[0] % config.size == 0, "Bloom array has incorrect size"
        bloom = BloomCounter(config)
        bloom._bloom_array = bloom_array
        return bloom

    def tobytes(self, n_partitions: int):
        """
        Serializes the bloom array into n_partitions files.
        """
        # Since we have fully materialized hash counter, no need to search with bin_search
        # indexes are [left, right)

        left_idx = 0
        bucket_size = self._bloom_array.size // n_partitions
        for bucket_i in range(n_partitions):
            # last bucket needs to have everything
            right_idx = (
                bucket_size * (bucket_i + 1)
                if bucket_i != n_partitions - 1
                else self._bloom_array.size
            )
            yield self._bloom_array[left_idx:right_idx].tobytes()
            left_idx = right_idx

    @classmethod
    def frombuffers(
        cls, buffers: Iterable[bytes], bloom_config: BloomCounterConfig
    ) -> "BloomCounter":
        """
        Deserializes partitions of an array into a single array.
        The size of the expected result must be known in time, so that we can preallocate the array
        which lowers memory consumption.

        The arrays from buffers are verticaly stacked, thus their total size must be equal to bloom_config.size
        """

        array = np.empty(bloom_config.size, dtype=bloom_config.dtype)

        left_idx = 0
        for buffer in buffers:
            loaded_array = np.frombuffer(buffer, dtype=bloom_config.dtype)
            array[left_idx : left_idx + loaded_array.shape[0]] = loaded_array
            left_idx += loaded_array.shape[0]

        assert (
            left_idx == bloom_config.size
        ), f"Expected to load {bloom_config.size} elements, but loaded {left_idx}"
        return cls.from_array(array, bloom_config)

    def get_indexes(self, shingles: np.ndarray):
        """Get indexes for the shingles with the k hashing functions"""
        return shingles % self._bloom_array.size

    def __add__(self, other: "BloomCounter") -> "BloomCounter":
        assert (
            self._bloom_array.shape == other._bloom_array.shape
        ), "Bloom counters have different sizes"
        new_count = self._bloom_array + other._bloom_array
        overflow_mask = new_count < self._bloom_array
        self._bloom_array = np.where(
            overflow_mask, np.iinfo(self._bloom_array.dtype).max, new_count
        )
        return self


class TopK:
    """
    TopK is a class that should support:
    1. fast min lookup
    2. relatively fast insert
    3. relatively fast removal of the min element
    It uses a heap to keep the k most frequent ngrams.
    """

    def __init__(self, k: int, ngram_size: int, count_dtype: np.dtype):
        # TODO: get the ngram_size only when saving/loading
        self.k = k
        self.heap: list[tuple[int, str]] = []
        # Use set for faster duplicate look-up
        self.elements = set()
        self.ngram_size = ngram_size
        self.count_dtype = count_dtype

    @property
    def min(self):
        return self.heap[0][0] if self.heap else -1

    def add(self, ngram: str, count: int):
        # First check that we don't have the ngram in the heap already
        if ngram in self.elements:
            return

        if len(self.heap) == self.k and self.min > count:
            # Then check if we have still space in heap
            return

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (count, ngram))
        else:
            # We are bigger than min
            removed_element = heapq.heapreplace(self.heap, (count, ngram))
            self.elements.remove(removed_element[1])

        self.elements.add(ngram)

    @classmethod
    def frombuffer(
        cls, buffer: bytes, k: int, ngram_size: int, count_dtype: np.dtype
    ) -> "TopK":
        topK = TopK(k, ngram_size, count_dtype)
        topK.heap = np.frombuffer(
            buffer, dtype=[("count", count_dtype), ("ngram", f"<U{ngram_size}")]
        ).tolist()
        topK.elements = set(ngram for ngram, _ in topK.heap)
        return topK

    def tobytes(self):
        # Not sure whether it's faster than poping and reinstering
        return np.array(
            self.heap,
            dtype=[("count", self.count_dtype), ("ngram", f"<U{self.ngram_size}")],
        ).tobytes()


class BloomCounterNgrams(PipelineStep):
    """
    Computes NGrams occurence, using BloomCounter and saves the BloomCounter to the sik

    Args:
        output_folder: folder where signatures + counters are saved
        workers: number of workers that will be used in bloomCounter merging stage
        config: configuration of the bloom filter
    """

    type = "🧐 - INSPECT"
    name = "🔤 top-k-ngrams-bloom-counter"
    _requires_dependencies = ["nltk", "mmh3"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        bloom_config: BloomCounterConfig,
        ngram_config: NgramsConfig,
        finder_workers: int = 1,
        buffer_size: int = 1000,
    ):
        super().__init__()

        if finder_workers <= 0:
            raise ValueError("finder_workers must be >= 1")
        elif finder_workers > 1:
            logger.warning(
                f"Remember to also set the name of tasks of the finder block to {finder_workers=}!"
            )

        self.finder_workers = finder_workers
        self.output_folder = get_datafolder(output_folder)
        self.bloom_config = bloom_config
        self.ngram_config = ngram_config
        self.buffer_size = buffer_size

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        bloomCounter = BloomCounter(self.bloom_config)
        with self.track_time():
            for doc in data:
                bloomCounter.add(get_ngrams(doc, self.ngram_config))

            for i, buffer in enumerate(bloomCounter.tobytes(self.finder_workers)):
                with self.output_folder.open(
                    f"{i:04d}/{rank:04d}{ExtensionHelperTN.stage_1_bc_local}", "wb"
                ) as f:
                    f.write(buffer)


class BloomCounterMerge(PipelineStep):
    """
    Merge bloom partitions into a single bloom counter.
    """

    type = "🧐 - INSPECT"
    name = "🔤 top-k-ngrams-bloom-merge"
    _requires_dependencies = ["nltk", "mmh3"]

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        bloom_config: BloomCounterConfig,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.bloom_config = bloom_config
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        files = self.input_folder.list_files(
            glob_pattern=f"{rank:04d}/*{ExtensionHelperTN.stage_1_bc_local}"
        )
        partition_size = self.bloom_config.size // world_size
        if rank == world_size - 1:
            partition_size = self.bloom_config.size - partition_size * (world_size - 1)

        if partition_size == 0 and rank == 0:
            raise ValueError("Number of workers is bigger than size of bloom filter")

        if partition_size <= 0:
            logger.warning(
                f"Partition size is too small for {rank=} and {world_size=}. Skipping."
            )
            return

        partition_bloom_config = replace(self.bloom_config, size=partition_size)

        bloom_counter = BloomCounter(partition_bloom_config)
        with self.track_time():
            for file in tqdm(self.input_folder.open_files(files)):
                bloom_counter += bloom_counter.frombuffers(
                    [file.read()], partition_bloom_config
                )

            with self.output_folder.open(
                f"{rank:04d}{ExtensionHelperTN.stage_2_bc_global}", "wb"
            ) as f:
                f.write(next(bloom_counter.tobytes(1)))


class TopKCounter(PipelineStep):
    """
    Computes TopK from a BloomCounter and saves them to disk.
    We can do that in this stage as we have the global BloomCounter,
    which guarantees that even though we take topK from different partitions,
    every global TopK will be in at least one of the partitions.
    """

    type = "🧐 - INSPECT"
    name = "🔤 top-k-ngrams-local-top"
    _requires_dependencies = ["nltk", "mmh3"]

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        bloom_config: BloomCounterConfig,
        ngram_config: NgramsConfig,
        k: int,
    ):
        self.bloom_folder = get_datafolder(input_folder)
        self.bloom_config = bloom_config
        self.ngram_config = ngram_config
        self.k: int = k
        self.output_folder = get_datafolder(output_folder)
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        files = sorted(self.bloom_folder.list_files(glob_pattern="*"))

        def read_file(file):
            with self.bloom_folder.open(file, "rb") as f:
                return f.read()

        bloom_counter = BloomCounter.frombuffers(
            (read_file(f) for f in files), self.bloom_config
        )

        topK = TopK(
            self.k,
            self.ngram_config.n,
            self.bloom_config.dtype,
        )
        # This time we do not counting, we only use bloom counter as refferece as now it's global one
        with self.track_time():
            for doc in data:
                # Could be sped up a bit by buffering the ngrams, since we vectorize stuff
                ngrams = list(get_ngrams(doc, self.ngram_config))
                ngrams_count = bloom_counter.get(ngrams)
                for ngram, count in zip(ngrams, ngrams_count):
                    topK.add(ngram, count)

            with self.output_folder.open(
                f"{rank:04d}{ExtensionHelperTN.stage_3_top_k_local}", "wb"
            ) as f:
                f.write(topK.tobytes())


class TopKMerge(PipelineStep):

    type = "🧐 - INSPECT"
    name = "🔤 top-k-ngrams-global-top"
    _requires_dependencies = ["nltk", "mmh3"]

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        k: int,
        bloom_config: BloomCounterConfig,
        ngram_config: NgramsConfig,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.k = k
        self.bloom_config = bloom_config
        self.ngram_config = ngram_config

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        # TODO: Rewrite this to be distributed
        if world_size != 1:
            raise ValueError("world_size must be 1 for this step")

        files = self.input_folder.list_files(
            glob_pattern=f"**/*{ExtensionHelperTN.stage_3_top_k_local}"
        )

        topK = TopK(self.k, self.ngram_config.n, self.bloom_config.dtype)
        with self.track_time():
            for f in self.input_folder.open_files(files):
                top_k_instance = TopK.frombuffer(
                    f.read(), self.k, self.ngram_config.n, self.bloom_config.dtype
                )
                for count, ngram in top_k_instance.heap:
                    topK.add(ngram, count)

            sorted_topK = sorted(topK.heap, key=lambda x: x[0])[: self.k]
            with self.output_folder.open(
                f"{rank:04d}{ExtensionHelperTN.stage_4_top_k_global}", "wt"
            ) as f:
                import csv

                writer = csv.writer(f)
                writer.writerows(sorted_topK)
