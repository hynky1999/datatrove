import os
import shutil
import tempfile
import unittest
import numpy as np
from datatrove.data import Document
from datatrove.pipeline.dedup.utils import ExtensionHelperSD

from tests.utils import require_nltk
from datatrove.pipeline.dedup.gram_inspector import GramFinderSignature, NgramMerge


TEXTS = ["hello", "hel", "lo", "hllo"]

GRAMS = [
    ("ell", 1),
    ("hel", 2),
    ("hll", 1),
    ("llo", 2),
]
    
    


class TestFilters(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_signature(self):
        individual_inspector = GramFinderSignature(self.tmp_dir)
        individual_inspector.run([Document(text=text, id=str(i)) for i, text in enumerate(TEXTS)])

        for file in individual_inspector.output_folder.list_files():
            content = individual_inspector.output_folder.open(file)
            arr = np.fromfile(content, dtype=[("ngram", f"<U3"), ("count", "<u8")])
            self.assertEqual(arr.tolist(), GRAMS)

    
    def test_merge(self):
        individual_inspector = NgramMerge(self.tmp_dir, self.tmp_dir)
        # create two files with ngrams
        FILE_1 = [
            ("ell", 1),
            ("hel", 1),
        ]

        FILE_2 = [
            ("hel", 1),
            ("hll", 1),
            ("llo", 2)
        ]

        for content, file in [(FILE_1, f"file1{ExtensionHelperSD.stage_1_signature}"), (FILE_2, f"file2{ExtensionHelperSD.stage_1_signature}")]:
            with individual_inspector.data_folder.open(file, "wb") as f:
                np.array(content, dtype=[("ngram", f"<U3"), ("count", "<u8")]).tofile(f)
        
        individual_inspector.run([])

        for file in individual_inspector.output_folder.glob(f"*{ExtensionHelperSD.stage_2_duplicates}"):
            with individual_inspector.output_folder.open(file) as f:
                arr = np.fromfile(f, dtype=[ ("count", "<u8"), ("ngram", f"<U3")])
                count_second = [(ngram, count) for count, ngram in arr.tolist()]
                # Check that ngrams have correct counts

                self.assertEqual(GRAMS, sorted(count_second, key=lambda x: x[0]))
                # Check that the arr is sorted by count
                self.assertEqual(arr["count"].tolist(), sorted(arr["count"].tolist(), reverse=True))
    

    

