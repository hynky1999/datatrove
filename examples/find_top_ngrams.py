from multiprocessing import freeze_support
import sys

import numpy as np
from datatrove.executor.local import LocalPipelineExecutor

from datatrove.executor.slurm import SlurmPipelineExecutor
from loguru import logger
from datatrove.pipeline.stats.gram_inspector import (
    BloomCounterConfig,
    BloomCounterMerge,
    BloomCounterNgrams,
    NgramsConfig,
    TopKCounter,
    TopKMerge,
)
from datatrove.pipeline.readers.jsonl import JsonlReader


# DUMP should be given as an argument. Example: CC-MAIN-2023-23
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)
DUMP = sys.argv[1]

LIMIT = 10000

# https://stats.stackexchange.com/questions/79365/number-of-ngrams-for-vocabulary
# Let's assume max is smth like 5_000_000_000 at 100 gram
EXPECTED_UNIQUE_NGRAMS = 5_000_000_000
# While we will likely not known the exact number uint64 will take twice the mem
DTYPE = np.uint32

LOGS_FOLDER = "/fsx/hynek_kydlicek/fineweb/experiments/ngrams"
OUTPUT_FOLDER = "s3://fineweb-data-processing-us-east-1/ngrams"

K = 1000

MERGE_WORKERS = 200

bloom_cfg = BloomCounterConfig.from_expected_elements(
    expected_elements=EXPECTED_UNIQUE_NGRAMS, expected_mem_gb=128, dtype=np.uint32
)

ngram_config = NgramsConfig(char_level=False, n=30)

logger.info(
    f"""
    Probability of incorrect count estimation: {bloom_cfg.prob_incorrect_count(EXPECTED_UNIQUE_NGRAMS):.2f}
    # Hash function: {bloom_cfg.n_hash_fcs}
"""
)


bloom_counter = [
    JsonlReader(text_key="content", progress=True, data_folder=DUMP, limit=LIMIT),
    BloomCounterNgrams(
        output_folder=f"{OUTPUT_FOLDER}/bloom",
        bloom_config=bloom_cfg,
        ngram_config=ngram_config,
        finder_workers=MERGE_WORKERS,
    ),
]

bloom_merge = BloomCounterMerge(
    input_folder=f"{OUTPUT_FOLDER}/bloom",
    output_folder=f"{OUTPUT_FOLDER}/bloom_merged",
    bloom_config=bloom_cfg,
)

top_k_inspector = [
    JsonlReader(data_folder=DUMP, text_key="content", progress=True, limit=LIMIT),
    TopKCounter(
        input_folder=f"{OUTPUT_FOLDER}/bloom_merged",
        output_folder=f"{OUTPUT_FOLDER}/top_k",
        bloom_config=bloom_cfg,
        ngram_config=ngram_config,
        k=K,
    ),
]

top_k_merge = TopKMerge(
    input_folder=f"{OUTPUT_FOLDER}/top_k",
    output_folder=f"{OUTPUT_FOLDER}/final",
    bloom_config=bloom_cfg,
    ngram_config=ngram_config,
    k=K,
)


executor_bloom_counter = LocalPipelineExecutor(
    pipeline=bloom_counter,
    logging_dir=LOGS_FOLDER,
    tasks=1,
)

executor_bloom_merge = LocalPipelineExecutor(
    depends=executor_bloom_counter,
    pipeline=[bloom_merge],
    logging_dir=LOGS_FOLDER,
    tasks=MERGE_WORKERS,
)

executor_top_k_inspector = LocalPipelineExecutor(
    depends=executor_bloom_merge,
    pipeline=top_k_inspector,
    logging_dir=LOGS_FOLDER,
    tasks=4,
)

executor_top_k_merge = LocalPipelineExecutor(
    depends=executor_top_k_inspector,
    pipeline=[top_k_merge],
    logging_dir=LOGS_FOLDER,
    tasks=1,
)

# Execute the pipeline
if __name__ == "__main__":
    freeze_support()
    executor_bloom_counter.run()
    executor_bloom_merge.run()
    executor_top_k_inspector.run()
    executor_top_k_merge.run()
