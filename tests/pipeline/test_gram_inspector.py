import csv
import os
import shutil
import tempfile
import unittest
import numpy as np
from datatrove.data import Document
from datatrove.io import DataFolder
from datatrove.pipeline.dedup.utils import ExtensionHelperTN

from tests.utils import require_nltk
from datatrove.pipeline.stats.gram_inspector import (
    BloomCounter,
    BloomCounterNgrams,
    BloomCounterMerge,
    BloomCounterConfig,
    NgramsConfig,
    TopKMerge,
    TopKCounter,
)


TEXTS_CHAR = [
    Document(text="hello", id="0"),
    Document(text="hel", id="1"),
    Document(text="lo", id="2"),
    Document(text="hllo", id="3"),
]

GRAMS_CHAR = [
    ("ell", 1),
    ("hel", 2),
    ("hll", 1),
    ("llo", 2),
]

TEXTS_WORDS = [
    Document(text="hello how are you?", id="0"),
    Document(text="how are you?", id="1"),
    Document(text="are you?", id="2"),
    Document(text="you?", id="3"),
]

GRAMS_WORDS = [
    ("hello how", 1),
    ("how are", 2),
    ("are you", 3),
    ("you ?", 4),
]


@require_nltk
class TestFilters(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_bloom_counter_step(self):
        config = BloomCounterConfig(dtype=np.uint32, n_hash_fcs=2, size=30)
        ngram_config = NgramsConfig(n=3, char_level=True)
        bloom_folder = DataFolder(f"{self.tmp_dir}/bloom")

        bloom_block = BloomCounterNgrams(
            output_folder=bloom_folder,
            bloom_config=config,
            ngram_config=ngram_config,
        )

        bloom_block.run(data=TEXTS_CHAR)

        bloom_files = bloom_folder.list_files(
            glob_pattern=f"**/*{ExtensionHelperTN.stage_1_bc_local}"
        )

        self.assertEqual(len(bloom_files), 1)
        bloom_array = BloomCounter.frombuffers(
            (f.read() for f in bloom_folder.open_files(bloom_files)),
            config,
        )

        counts = bloom_array.get([g for g, _ in GRAMS_CHAR])
        self.assertEqual(counts, [c for _, c in GRAMS_CHAR])

    def test_bloom_overflow(self):
        config = BloomCounterConfig(dtype=np.uint32, n_hash_fcs=2, size=10)
        bloom_array = BloomCounter(config)
        bloom_array._bloom_array += np.iinfo(np.uint32).max - 2
        bloom_array.add(list("a" * 100))
        # Check that we haven't overflown
        self.assertEqual(bloom_array.get("a")[0], np.iinfo(np.uint32).max)

    def test_bloom_add_return(self):
        config = BloomCounterConfig(dtype=np.uint32, n_hash_fcs=2, size=30)
        bloom_array = BloomCounter(config)
        counts = bloom_array.add("abcd")
        self.assertEqual(counts, [1, 1, 1, 1])
        counts = bloom_array.add("abcd")
        self.assertEqual(counts, [2, 2, 2, 2])
        counts = bloom_array.add("a")
        self.assertEqual(counts, [3])

    def test_bloom_distributed_merge(self):
        config = BloomCounterConfig(dtype=np.uint32, n_hash_fcs=2, size=30)

        ngram_config = NgramsConfig(n=3, char_level=True)
        bloom_folder = DataFolder(f"{self.tmp_dir}/bloom")
        bc1 = BloomCounterNgrams(
            output_folder=bloom_folder,
            bloom_config=config,
            ngram_config=ngram_config,
            finder_workers=4,
        )

        merge_block = BloomCounterMerge(
            input_folder=bloom_folder,
            output_folder=bloom_folder,
            bloom_config=config,
        )

        bc1.run(data=TEXTS_CHAR[:2], rank=0, world_size=2)
        bc1.run(data=TEXTS_CHAR[2:], rank=1, world_size=2)

        for i in range(4):
            merge_block.run(data=TEXTS_CHAR, rank=i, world_size=4)

        bloom_files = bloom_folder.list_files(
            glob_pattern=f"**/*{ExtensionHelperTN.stage_2_bc_global}"
        )

        self.assertEqual(len(bloom_files), 4)
        bloom_array = BloomCounter.frombuffers(
            (f.read() for f in bloom_folder.open_files(bloom_files)),
            config,
        )

        counts = bloom_array.get([g for g, _ in GRAMS_CHAR])
        self.assertEqual(counts, [c for _, c in GRAMS_CHAR])

    def test_topk_ngrams(self):
        for text, grams, char_level, n, name in [
            (TEXTS_WORDS, GRAMS_WORDS, False, 2, "bloom-words"),
            (TEXTS_CHAR, GRAMS_CHAR, True, 3, "bloom-char"),
        ]:
            config = BloomCounterConfig(dtype=np.uint32, n_hash_fcs=2, size=20)

            ngram_config = NgramsConfig(n=n, char_level=char_level)
            bloom_folder = DataFolder(f"{self.tmp_dir}/{name}")
            bc = BloomCounterNgrams(
                output_folder=bloom_folder,
                bloom_config=config,
                ngram_config=ngram_config,
                finder_workers=1,
            )

            merge_block = BloomCounterMerge(
                input_folder=bloom_folder,
                output_folder=bloom_folder,
                bloom_config=config,
            )

            k = 2
            topK = TopKCounter(
                input_folder=bloom_folder,
                output_folder=bloom_folder,
                bloom_config=config,
                ngram_config=ngram_config,
                k=k,
            )
            topKMerge = TopKMerge(
                input_folder=bloom_folder,
                output_folder=bloom_folder,
                bloom_config=config,
                ngram_config=ngram_config,
                k=k,
            )

            bc.run(data=text)
            merge_block.run(data=text)
            topK.run(data=text)
            topKMerge.run(data=text)

            with bloom_folder.open(
                f"{0:04d}{ExtensionHelperTN.stage_4_top_k_global}", "rt"
            ) as f:
                csv_reader = csv.reader(f)
                top_k = [(g, int(c)) for c, g in csv_reader]

            top_k_grams = sorted(grams, key=lambda x: x[1], reverse=True)[:k]
            self.assertEqual(set(top_k_grams), set(top_k_grams))
