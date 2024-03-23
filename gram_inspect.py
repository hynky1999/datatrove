from multiprocessing import freeze_support
import sys
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.gram_inspector import GramFinderSignature, NgramMerge
from datatrove.pipeline.readers import JsonlReader



step1 = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(
            data_folder="./random_text_records.jsonl"
        ),
        GramFinderSignature(
            finder_workers=1,
            output_folder="./counts"
        ),
    ],
    tasks=4,
)


step2 = LocalPipelineExecutor(
    depends=step1,
    pipeline=[
        NgramMerge(
            data_folder="./counts",
            lines_to_buffer=100,
            output_folder="./out",
        )
    ],
    tasks=1,
)

if __name__ == '__main__':
    freeze_support()
    step2.run()

