import json
from typing import Tuple
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.utils.stats import MetricStatsDict

SHORT_WORD_MAX_CHARS = 3


class WordStats(PipelineStep):
    type = "📊 - STATS"
    name = "_ word stats"
    _requires_dependencies = ["nltk", "tldextract"]


    def __init__(
        self,
        output_folder: DataFolderLike,
        round_digits: int = 3,
    ) -> None:
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.round_digits = round_digits


    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        from nltk import word_tokenize
        from tldextract import tldextract

        # Define a function to initialize counters
        def init_counters():
            return {
                "docs_per": MetricStatsDict(),
                "length_counter": MetricStatsDict(),
                "word_count": MetricStatsDict(),
                "avg_word_length_per": MetricStatsDict(),
                "white_space_counter": MetricStatsDict(),
                "white_space_ratio_counter": MetricStatsDict(),
                "short_word_counter": MetricStatsDict(),
                "short_word_ratio_counter": MetricStatsDict(),
                "non_alpha_digit_ratio": MetricStatsDict(),
            }

        # Initialize counters for both fqdn and domain suffix using the function
        default_counter = init_counters()
        fqdn_counters = init_counters()
        suffix_counters = init_counters()

        for doc in data:
            url_extracted = tldextract.extract(doc.metadata.get("url"))
            fqdn = url_extracted.fqdn
            suffix = url_extracted.suffix

            length = len(doc.text)

            words = word_tokenize(doc.text)
            word_count = len(words)
            avg_word_length = round(length / word_count, self.round_digits)

            white_space_count = sum(1 for word in words if word.isspace())
            white_space_ratio = round(white_space_count / len(words), self.round_digits)
            short_word_count = sum(1 for word in words if len(word) < SHORT_WORD_MAX_CHARS)
            short_word_ratio = round(short_word_count / len(words), self.round_digits)

            non_alpha_digit_ratio = round(sum(1 for word in words if not word.isalnum()) / len(words), self.round_digits)
            




            for counter, key in zip([default_counter, fqdn_counters, suffix_counters], [None, fqdn, suffix]):
                if key is None:
                    counter["docs_per"]["default"] += 1
                    counter["length_counter"][length] += 1
                    counter["word_count"][word_count] += 1
                    counter["avg_word_length_per"][avg_word_length] += 1
                    counter["white_space_counter"][white_space_count] += 1
                    counter["white_space_ratio_counter"][white_space_ratio] += 1
                    counter["short_word_counter"][short_word_count] += 1
                    counter["short_word_ratio_counter"][short_word_ratio] += 1
                    counter["non_alpha_digit_ratio"][non_alpha_digit_ratio] += 1


                else:
                    counter["docs_per"][key] += 1
                    counter["length_counter"][key] += length
                    counter["word_count"][key] += word_count
                    counter["avg_word_length_per"][key] += avg_word_length
                    counter["white_space_counter"][key] += white_space_count
                    counter["white_space_ratio_counter"][key] += white_space_ratio
                    counter["short_word_counter"][key] += short_word_count
                    counter["short_word_ratio_counter"][key] += short_word_ratio
                    counter["non_alpha_digit_ratio"][key] += non_alpha_digit_ratio

            yield doc

        # save to disk
        with self.output_folder.open(f"{rank:05d}_word_stats.json", "wt") as f:
            json.dump(
            {
                "document": {key: value.to_dict() for key, value in default_counter.items()},
                "fqdn": {key: value.to_dict() for key, value in fqdn_counters.items()},
                "suffix": {key: value.to_dict() for key, value in suffix_counters.items()},
            },
            f
            )