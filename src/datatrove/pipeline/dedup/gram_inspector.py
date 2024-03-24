from collections import defaultdict
import heapq
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import struct
from typing import Callable, Generator

import numpy as np
from fsspec.spec import AbstractBufferedFile
from loguru import logger
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from .utils import ExtensionHelperSD


@dataclass
class GramFinderConfig:
    ngram: int = 3
    is_relevant: Callable[[str], bool] = lambda x: True


DEFAULT_GRAM_FINDER_CONFIG = GramFinderConfig()


@dataclass(order=True)
class Sig:
    n_gram: str
    count: int
    file_id: int


class GramFinderSignature(PipelineStep):
    """SentenceDedup: First pipeline step

        Creates a signature for each sentence in each document. Each HashSig has n hash, the doc id and the sentence idx. Before saving
        them the hashes are sorted. 

    Args:
        output_folder: folder where signatures are saved
    """

    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 1"
    _requires_dependencies = ["nltk"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        finder_workers: int = 1,
        config: GramFinderConfig = DEFAULT_GRAM_FINDER_CONFIG,
        language: str = "english",
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        if finder_workers <= 0:
            raise ValueError("finder_workers must be >= 1")
        elif finder_workers > 1:
            logger.warning(f"Remember to also set the name of tasks of the finder block to {finder_workers=}!")
        self.finder_workers = finder_workers
        self.config = config
        self.language = language

    def save(self, rank: int, signatures):
        # explicitly define little endiannes
        signatures = np.array(signatures, dtype=[("ngram", f"<U{self.config.ngram}"), ("count", "<u8")])
        signatures = np.sort(signatures)
        with self.output_folder.open(f"{rank:04d}{ExtensionHelperSD.stage_1_signature}", "wb") as f:
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
        counter = defaultdict(int)
        for doc in tqdm(data):
            with self.stats.time_stats:
                for n_gram in self.get_ngrams(doc):
                    counter[n_gram] += 1

        
        counts = sorted(counter.items(), key=lambda x: x[0])
        for (ngram, count) in counts:
            if count == 17:
                print(ngram, count)
        self.save(rank, counts)

class NgramMerge(PipelineStep):
    """NgramMerge: Second pipeline step

        SentenceFindDedups runs on a single worker. It reads all the signatures from the previous step and loads them
        in a priority queue to check for duplicates. If a duplicate is found its document id and sentence id are saved.

    Args:
        data_folder: data folder where signatures are saved
        output_folder: folder where duplicates are saved
        index_folder: folder where index files are saved
        only_dedup_in_index: only dedup in index
    """

    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 2"

    def __init__(
        self,
        data_folder: DataFolderLike,
        output_folder: DataFolderLike,
        config: GramFinderConfig = DEFAULT_GRAM_FINDER_CONFIG,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.output_folder = get_datafolder(output_folder)
        self.config = config
        self.records_to_buffer = lines_to_buffer

    def read_sigs(self, file: AbstractBufferedFile, file_id, records_to_buffer: int = 5
    ) -> Generator[Sig, None, None]:
        reader = struct.Struct(f"<{self.config.ngram * 'i'}Q")
        with file as f:
            while True:
                records = f.read(records_to_buffer * reader.size)
                if not records:
                    return

                parsed_records = np.frombuffer(records, dtype=[("ngram", f"<U{self.config.ngram}"), ("count", "<u8")])
                for record in parsed_records:
                    yield Sig(record["ngram"], record["count"], file_id)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        if world_size != 1:
            raise ValueError()
        with self.stats.time_stats:
            sig_files = self.data_folder.list_files(glob_pattern="*" + ExtensionHelperSD.stage_1_signature)
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
            out_filename = f"top_{self.config.ngram}.{ExtensionHelperSD.stage_2_duplicates}"

            counts = []


            while pq:
                v: Sig = heapq.heappop(pq)
                if last and last.n_gram == v.n_gram:
                    last.count += v.count
                else:
                    if last:
                        assert last.count < 2**(8*8), "Ngram count overflow"
                        counts.append(np.array([(last.count, last.n_gram)], dtype=[("count", "<u8"), ("ngram", f"<U{self.config.ngram}")]))
                    last = v

                new_v = next(sig_readers[v.file_id], None)

                if new_v:
                    heapq.heappush(pq, new_v)
            
            if last:
                assert last.count < 2**(8*8), "Ngram count overflow"
                counts.append(np.array([(last.count, last.n_gram)], dtype=[("count", "<u8"), ("ngram", f"<U{self.config.ngram}")]))

    
        # now load output_mg inmemory and sort it based on the count
        arr = np.sort(np.concatenate(counts, dtype=[("count", "<u8"), ("ngram", f"<U{self.config.ngram}")]), order="count")[::-1]


        # now save it to the output_mg
        with self.output_folder.open(out_filename, mode="wb") as f:
            f.write(arr.tobytes())