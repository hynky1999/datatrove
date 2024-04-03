from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import heapq
from logging import Logger
from typing import Generator
import numpy as np
from tqdm import tqdm
from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder

from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.utils import ExtensionHelperTN
from fsspec.spec import AbstractBufferedFile
from loguru import logger
from datatrove.pipeline.stats.gram_inspector import (
    BloomCounter,
    BloomCounterConfig,
    NgramsConfig,
    TopK,
    get_ngrams,
)


@dataclass(order=True)
class NgramCount:
    ngram: str
    count: int
    file_id: int = 0


class TopKSortedArray:
    """
    TopKSortedArray is a class that implements non-probabilistic TopK strucutre.
    Assumptions:
    - The values of the same elements added increasing by 1
    Complexity:
    add: O(1)

    Because we know that inserted elements are always gonna be +1 compared to previous insert,
    using sorted array is faster than heap.
    1) The min is always at the end O(1)
    2) We know that if the element in not in the heap it's at least by 1 smaller compared to min, so that when we
    finally insert the element, we can just replace the min element by the new one. This make the insert O(1)
    3) We know that when we increment the element in the list, we will always swap at most 1 element.
    """

    MASK = (0, "")

    def __init__(self, k: int, count_dtype: np.dtype):
        # TODO: get the ngram_size only when saving/loading
        self.k = k
        self.sorted_arr: list[tuple[int, str]] = [self.MASK] * k
        self.index_map: dict[str, int] = {}
        self.count_dtype = count_dtype

    def __len__(self):
        return len(self.index_map)

    @property
    def min(self):
        return (
            self.sorted_arr[len(self) - 1][0]
            if len(self) > 0
            else np.iinfo(self.count_dtype).min
        )

    def _update(self, ngram: str, count):
        index = self.index_map[ngram]
        self.sorted_arr[index] = (count, ngram)
        self._swap_up(index)

    def _swap_up(self, index: int):
        """
        Bubble up the element at index until it's ge than its predecessor
        """

        elem = self.sorted_arr[index]
        value = elem[0]
        while index > 0 and self.sorted_arr[index - 1][0] < value:
            # Small optimization if we have sequence of the same values,
            # we have to only swap with the last element
            l_end = self._get_left_end_of_same_value_sequence(index - 1)
            # Move the end to index
            self.sorted_arr[index] = self.sorted_arr[l_end]
            self.index_map[self.sorted_arr[index][1]] = index

            index = l_end

        self.sorted_arr[index] = elem
        self.index_map[elem[1]] = index
        return index

    def _get_left_end_of_same_value_sequence(self, index: int):
        while index > 0 and self.sorted_arr[index - 1][0] == self.sorted_arr[index][0]:
            index -= 1
        return index

    def _try_insert(self, ngram: str, count: int):
        # We are full and the new element is smaller than the min
        if len(self) == self.k and count < self.min:
            return

        # We are not full, we simply replace the mask element
        if len(self) < self.k:
            self.sorted_arr[len(self)] = (count, ngram)
            self.index_map[ngram] = len(self)
            self._swap_up(len(self) - 1)

        # We are full, but the new element is bigger than the min
        elif count > self.min:
            del self.index_map[self.sorted_arr[len(self) - 1][1]]
            self.sorted_arr[len(self)] = (count, ngram)
            self.index_map[ngram] = len(self)
            self._swap_up(len(self) - 1)

    def add(self, ngram: str, count: int):
        if ngram in self.index_map:
            self._update(ngram, count)
        else:
            self._try_insert(ngram, count)

    def to_list(self):
        return self.sorted_arr[: len(self.index_map)]


class OptimisticBloomAndTopKCounter(PipelineStep):
    """
    Computes NGrams occurence, using BloomCounter and saves the BloomCounter to the sik.
    It calculates topk on the fly without saving the bloom counter. This might produce incorrect results

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
        k: int,
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
        self.k = k

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        bloomCounter = BloomCounter(self.bloom_config)
        topK = TopKSortedArray(self.k, self.bloom_config.dtype)
        with self.track_time():
            for doc in data:
                ngrams = list(get_ngrams(doc, self.ngram_config))
                totals = bloomCounter.add(ngrams)
                for total, ngram in zip(totals, ngrams):
                    topK.add(ngram, total)

            # Now we save the topK by sorting the array by ngrams
            sortedTopK = sorted(topK.to_list(), key=lambda x: x[1])
            with self.output_folder.open(
                mode="wb", path=f"{rank:04d}{ExtensionHelperTN.stage_3_top_k_local}"
            ) as f:
                max_char_length = max(len(ngram) for _, ngram in sortedTopK)
                data_array = np.array(
                    sortedTopK,
                    dtype=[
                        ("count", self.bloom_config.dtype),
                        ("ngram", f"<U{max_char_length}"),
                    ],
                )
                max_length_info = np.array([max_char_length], dtype=np.int64)
                f.write(max_length_info.tobytes() + data_array.tobytes())


def read_np_from_file(file: AbstractBufferedFile, dtype, lines_to_buffer: int = 5):
    """Utility which reads buffered data from a file and returns a numpy array.
    It doesn't use np.fromfile, because not all fsspec implementations support it.
    """
    size = np.dtype(dtype).itemsize * lines_to_buffer
    return np.frombuffer(file.read(size), dtype=dtype)


class MergeTopKCounters(PipelineStep):
    """ """

    type = "🧐 - INSPECT"
    name = "🔤 top-k-ngrams-bloom-counter"
    _requires_dependencies = ["nltk", "mmh3"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        input_folder: DataFolderLike,
        bloom_config: BloomCounterConfig,
        k: int,
    ):
        super().__init__()

        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.k = k
        self.bloom_config = bloom_config

    def read_sigs(
        self,
        file: AbstractBufferedFile,
        file_id: int,
        lines_to_buffer: int = 5,
    ) -> Generator[NgramCount, None, None]:
        last = None
        with file as f:
            # Read the size
            max_char_length = np.frombuffer(
                f.read(np.int64().itemsize), dtype=np.int64, count=1
            )[0]
            for data in read_np_from_file(
                f,
                [("count", self.bloom_config.dtype), ("ngram", f"<U{max_char_length}")],
                lines_to_buffer=lines_to_buffer,
            ):
                assert (
                    last is None or data[1] >= last
                ), f"Sort error. {f.tell()=}, {data[1]=}, {last=}"
                last = data[1]
                yield NgramCount(ngram=data[1], count=data[0], file_id=file_id)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        files = self.input_folder.list_files(
            glob_pattern=f"*{ExtensionHelperTN.stage_3_top_k_local}"
        )
        sig_readers = [
            self.read_sigs(file, file_id=file_id)
            for file_id, file in enumerate(self.input_folder.open_files(files))
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
        topK = TopK(self.k, self.bloom_config.dtype)
        last: NgramCount | None = None
        while pq:
            v: NgramCount = heapq.heappop(pq)
            if last and last.ngram == v.ngram:
                v.count += last.count

            elif last and last.ngram != v.ngram:
                topK.add(last.ngram, last.count)
            last = v

            next_v = next(sig_readers[v.file_id], None)
            if next_v:
                heapq.heappush(pq, next_v)

        if last:
            topK.add(last.ngram, last.count)

        with self.output_folder.open(
            mode="wt", path=f"{rank:04d}{ExtensionHelperTN.stage_4_top_k_global}"
        ) as output_mg:
            csv_writer = csv.writer(output_mg)
            for count, ngram in topK.heap:
                csv_writer.writerow([count, ngram])
