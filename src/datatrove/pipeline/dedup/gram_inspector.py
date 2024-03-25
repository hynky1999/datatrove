from collections import defaultdict
import hashlib
import heapq
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import struct
from typing import Any, Callable, Generator, Literal, TypedDict

import numpy as np
from fsspec.spec import AbstractBufferedFile
from loguru import logger
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from .utils import ExtensionHelperSD
import mmh3

_mersenne_prime = np.uint64((1 << 61) - 1)
MAX_HASH = 1 << 60 - 1


def mmh3_hash64(text: str) -> int:
    return mmh3.hash64(text)

def get_optimal_k(size_in_bytes: int, expected_elements: int) -> int:
    assert expected_elements, f"if {expected_elements=} then k must be given"
    m = size_in_bytes * 8
    k = (m / expected_elements) * np.log(2)
    return math.ceil(k)



@dataclass
class BloomFilterConfig:
    k: int
    counter_size: np.int8 | np.int32
    max_size: int = 1000  # in bits
    seed: int = 0


@dataclass
class GramFinderConfig:
    bloom_filter_config: BloomFilterConfig
    ngram: int = 3
    is_relevant: Callable[[str], bool] = lambda x: True


DEFAULT_GRAM_FINDER_CONFIG = GramFinderConfig()


@dataclass(order=True)
class Sig:
    n_gram: str
    count: int
    file_id: int


# Idea:
#  1. Have a bloom filter counter for each ngram
# 2. Have a topk, which saves also the ngram


# 1. Phase
# Compute BlomCounters

# 2. Phase
# Merge BloomCounters

# 3. Phase
# Given BloomCounter compute TopK

# 4. Phase
# Merge TopK


class NgramWithCount(TypedDict):
    ngram: str
    count: int


class BloomCounter:
    """
    Implementation of a bloom filter counter.
    Probability of collisions = (1 - e^(-n/m))^k
    https://en.wikipedia.org/wiki/Bloom_filter

    n = number of inserted elements
    m = size of the filter in bits
    k = number of hash functions (optimal = m/n * ln(2))
    """

    def __init__(self, config: BloomFilterConfig)
        self.config = config
        self.bloom_size = self.config.max_size // self.config.counter_size.itemsize
        self.seed = config.seed
        self.k = config.k
        self.bloom_array = np.zeros(
            self.bloom_size,
            dtype=self.config.counter_size,
        )


    def add(self, ngrams: list[str]):
        """
        Adds an gram to the bloom filter and returns it's approximate count
        """

        shingles = np.array([mmh3_hash64(ngram) for ngram in ngrams])
        indexes = self.get_indexes(shingles)
        self.bloom_array[indexes] += 1


    def serialize(self, file: AbstractBufferedFile):
        file.write(self.bloom_array.tobytes())

    def from_array(self, bloom_array: np.ndarray, config: BloomFilterConfig) -> "BloomCounter":
        assert bloom_array.shape[0] % config.max_size == 0, "Bloom array not fully initialized"
        bloom = BloomCounter(config)
        bloom.bloom_array = bloom_array
        return bloom

    @property
    def parameters(self):
        """Returns the parameters for the hash functions.
            Create parameters for a random bijective permutation function
            that maps a 32-bit hash value to another 32-bit hash value.
            http://en.wikipedia.org/wiki/Universal_hashing

        Returns:
            tuple: (a, b) parameters for the hash functions
                where a and b are numpy uint64 arrays of shape (1, k) containing the
                random parameters for the hash functions.
        """
        if not self._parameters:
            gen = np.random.RandomState(self.seed)
            self._parameters = (
                gen.randint(1, _mersenne_prime, dtype=np.uint64, size=(1, self.k)),
                gen.randint(0, _mersenne_prime, dtype=np.uint64, size=(1, self.k)),
            )
        return self._parameters

    def get_indexes(self, shingles: np.ndarray) -> list[list[int]]:
        """Get indexes for the shingles with the k hashing functions"""
        a, b = self.parameters
        phv = np.bitwise_and((shingles * a + b) % _mersenne_prime, self.config.max_size)
        return phv.tolist()

    @classmethod
    def from_file(cls, file: AbstractBufferedFile, config: BloomFilterConfig) -> "BloomCounter":
        return cls.from_files([file], config)


    @classmethod
    def _deserialize(cls, file: AbstractBufferedFile, config: BloomFilterConfig, buffered_records: int = 300) -> Generator[tuple[np.ndarray[np.int8 | np.int32, Any]], None, None]:
        while True:
            records = file.read(buffered_records * config.counter_size.itemsize)
            if not records:
                return
            parsed_records = np.frombuffer(records, dtype=config.counter_size)
            yield parsed_records 

    @classmethod
    def from_files(cls, files: list[AbstractBufferedFile], config: BloomFilterConfig, buffered_records: int = 300) -> "BloomCounter":
        """
        Merge multiple bloom counters into a single bloom counter.
        """

        def yield_partials(files: list[AbstractBufferedFile], config: BloomFilterConfig) -> Generator[list[np.ndarray], None, None]:
            deserialize_iterators = [cls._deserialize(file, config, buffered_records=buffered_records) for file in files]
            with ThreadPoolExecutor() as executor:
                while True:
                    partials = list(executor.map(lambda x: next(x, None), deserialize_iterators))
                    if any(partial is None for partial in partials):
                        assert all(partial is None for partial in partials), "Not all bloom counters are same length"
                        return

                    yield partials

        bloom = BloomCounter(config)
        offset = 0
        # compute total based on config and size of counter
        for blooms_parts in tqdm(yield_partials(files, config))
            bloom.bloom_array[offset:offset + blooms_parts[0].shape[0]] = np.sum(blooms_parts, axis=1)
            offset += blooms_parts[0].shape[0]
        
        assert offset == bloom.bloom_array.shape[0], "Bloom array not fully initialized"
        return bloom


class TopK:
    """
    TopK is a class that implements a priority queue of the k most frequent ngrams.
    It uses a heap to keep the k most frequent ngrams.
    """

    def __init__(self, k: int):
        self.k = k
        self.heap = []
        self.index_map = {}


    @property
    def get_min(self):
        return self.heap[0] if self.heap else 0

    def _insert(self, ngram, count: int):
        heapq.heapreplace
        self.index_map[ngram] = len(self.heap) - 1



    def add(self, ngram: NgramWithCount, count: int):
        index = self.index_map.get(ngram, None)
        if index is None:
            # New element
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, (count, ngram))
                self.index_map[ngram] = len(self.heap) - 1
            else:
                if self.get_min < count:
                    heapq.heappop(self.heap)
                
                



            




            
        


        

    def remove(self, ngram: str):
        pass

    def from_files(self, files: list[AbstractBufferedFile]) -> "TopK":
        """
        Merge multiple topk into a single topk.
        """
        pass

    def serialize(self, file: AbstractBufferedFile):
        pass


class BloomCounterNgrams(PipelineStep):
    """
    Computes NGrams occurence, using BloomCounter and saves them to disk.

    Args:
        output_folder: folder where signatures are saved
        config: configuration of the bloom filter

    """

    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 1"
    _requires_dependencies = ["nltk"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        config: GramFinderConfig = DEFAULT_GRAM_FINDER_CONFIG,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.config = config

    def save(self, rank: int, signatures):
        # explicitly define little endiannes
        signatures = np.array(
            signatures, dtype=[("ngram", f"<U{self.config.ngram}"), ("count", "<u8")]
        )
        signatures = np.sort(signatures)
        with self.output_folder.open(
            f"{rank:04d}{ExtensionHelperSD.stage_1_signature}", "wb"
        ) as f:
            f.write(signatures.tobytes())

    def get_ngrams(self, doc: Document):
        from nltk import ngrams

        tokens = [x for x in doc.text if self.config.is_relevant(x)]
        n_grams = ["".join(ngram) for ngram in ngrams(tokens, self.config.ngram)]
        return n_grams

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        """Args:
            data
            rank
            world_size

        Returns:

        SentenceDedupSignature creates a signature for each document. Each HashSig has n hash, the doc id and the
        sentence idx. Before saving them the hashes are sorted.
        """

        bloomCounter = BloomCounter(self.config.bloom_filter_config)
        for ngram in (ngram for doc in tqdm(data) for ngram in self.get_ngrams(doc)):
            bloomCounter.add(ngram)

        bloomCounter.serialize(
            self.output_folder.open(
                f"{rank:04d}{ExtensionHelperSD.stage_1_signature}", "wb"
            )
        )


class BloomCounterMerge(PipelineStep):
    """
    Merge BloomCounters from multiple files into a single BloomCounter.
    """

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):

        BloomCounter.from_array(bloom_array).serialize(
            self.output_folder.open(
                f"{rank:04d}{ExtensionHelperSD.stage_1_signature}", "wb"
            )
        )


class TopKCounter(PipelineStep):
    """
    Computes TopK from a BloomCounter and saves them to disk.
    """

    def __init__(self, merged_bloom_file: str, gram_finder_config: GramFinderConfig):
        self.merged_bloom_file = merged_bloom_file
        self.gram_finder_config = gram_finder_config
        self.output_folder = output_folder
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        bloomCounter = BloomCounter.from_file(self.merged_bloom_file)
        topK = TopK(
            self.gram_finder_config.ngram,
            self.output_folder.open(
                f"{rank:04d}{ExtensionHelperSD.stage_1_signature}", "wb"
            ),
        )

        for ngram, count in bloomCounter.get_topk(self.gram_finder_config.ngram):
            topK.add(ngram, count)

        topK.serialize(
            self.output_folder.open(
                f"{rank:04d}{ExtensionHelperSD.stage_1_signature}", "wb"
            )
        )


class TopKMerge(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 2"

    def __init__(
        self,
        data_folder: DataFolderLike,
        config: GramFinderConfig = DEFAULT_GRAM_FINDER_CONFIG,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.config = config
        self.records_to_buffer = lines_to_buffer

    def read_sigs(
        self, file: AbstractBufferedFile, file_id, records_to_buffer: int = 5
    ) -> Generator[Sig, None, None]:
        reader = struct.Struct(f"<{self.config.ngram * 'i'}Q")
        with file as f:
            while True:
                records = f.read(records_to_buffer * reader.size)
                if not records:
                    return

                parsed_records = np.frombuffer(
                    records,
                    dtype=[("ngram", f"<U{self.config.ngram}"), ("count", "<u8")],
                )
                for record in parsed_records:
                    yield Sig(record["ngram"], record["count"], file_id)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        if world_size != 1:
            raise ValueError()
        with self.stats.time_stats:
            sig_files = self.data_folder.list_files(
                glob_pattern="*" + ExtensionHelperSD.stage_1_signature
            )
            sig_readers = [
                self.read_sigs(file, file_id, records_to_buffer=self.records_to_buffer)
                for file_id, file in enumerate(self.data_folder.open_files(sig_files))
            ]

            logger.info(f"Initializing pq with {len(sig_readers)} files.")
            with ThreadPoolExecutor() as executor:
                pq = [
                    x
                    for x in tqdm(
                        executor.map(lambda x: next(x, None), sig_readers),
                        total=len(sig_readers),
                        desc="Initializing pq...",
                    )
                    if x
                ]
            heapq.heapify(pq)

            last: Sig | None = None
            out_filename = (
                f"top_{self.config.ngram}.{ExtensionHelperSD.stage_2_duplicates}"
            )

            counts = []

            while pq:
                v: Sig = heapq.heappop(pq)
                if last and last.n_gram == v.n_gram:
                    last.count += v.count
                else:
                    if last:
                        assert last.count < 2 ** (8 * 8), "Ngram count overflow"
                        counts.append(
                            np.array(
                                [(last.count, last.n_gram)],
                                dtype=[
                                    ("count", "<u8"),
                                    ("ngram", f"<U{self.config.ngram}"),
                                ],
                            )
                        )
                    last = v

                new_v = next(sig_readers[v.file_id], None)

                if new_v:
                    heapq.heappush(pq, new_v)

            if last:
                assert last.count < 2 ** (8 * 8), "Ngram count overflow"
                counts.append(
                    np.array(
                        [(last.count, last.n_gram)],
                        dtype=[("count", "<u8"), ("ngram", f"<U{self.config.ngram}")],
                    )
                )

        # now load output_mg inmemory and sort it based on the count
        arr = np.sort(
            np.concatenate(
                counts, dtype=[("count", "<u8"), ("ngram", f"<U{self.config.ngram}")]
            ),
            order="count",
        )[::-1]

        # now save it to the output_mg
        with self.o.open(out_filename, mode="wb") as f:
            f.write(arr.tobytes())
