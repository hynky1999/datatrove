import json
from typing import Tuple
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.utils.stats import MetricStatsDict

SHORT_LINE_MAX_CHARS = 35
SHORT_PARAGRAPH_MAX_LINES = 3


class LineStats(PipelineStep):
    type = "📊 - STATS"
    name = "_ line stats"
    _requires_dependencies = ["nltk", "tldextract"]


    def __init__(
        self,
        output_folder: DataFolderLike,
        round_digits: int = 3,
        short_line_max_chars: int | list[int] = SHORT_LINE_MAX_CHARS,
        short_paragraph_max_lines: int | list[int] = SHORT_PARAGRAPH_MAX_LINES,
    ) -> None:
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.round_digits = round_digits
        self.short_line_max_chars = short_line_max_chars if isinstance(short_line_max_chars, list) else [short_line_max_chars]
        self.short_paragraph_max_lines = short_paragraph_max_lines if isinstance(short_paragraph_max_lines, list) else [short_paragraph_max_lines]


    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        from nltk import word_tokenize
        from tldextract import tldextract

        # Define a function to initialize counters
        def init_counters():
            return {
                "docs_per": MetricStatsDict(),
                "line_count_per": MetricStatsDict(),
                "avg_line_length_per": MetricStatsDict(),
                "avg_words_per_line_counter": MetricStatsDict(),
                "paragraphs_num_counter": MetricStatsDict(),
                "paragraphs_avg_length_counter": MetricStatsDict(),
                **{
                    f"short_line_counter_{short_line_max}": MetricStatsDict()
                    for short_line_max in self.short_line_max_chars
                },
                **{
                    f"short_line_ratio_counter_{short_line_max}": MetricStatsDict()
                    for short_line_max in self.short_line_max_chars
                },
                **{
                    f"short_paragraphs_counter_{short_paragraph_max}": MetricStatsDict()
                    for short_paragraph_max in self.short_paragraph_max_lines
                },
                **{
                    f"short_paragraphs_ratio_counter_{short_paragraph_max}": MetricStatsDict()
                    for short_paragraph_max in self.short_paragraph_max_lines
                },
            }

        # Initialize counters for both fqdn and domain suffix using the function
        default_counter = init_counters()
        fqdn_counters = init_counters()
        suffix_counters = init_counters()

        for doc in data:
            url_extracted = tldextract.extract(doc.metadata.get("url"))
            fqdn = url_extracted.fqdn
            suffix = url_extracted.suffix

            words = word_tokenize(doc.text)
            lines = doc.text.splitlines()
            paragraphs = doc.text.split("\n\n")

            avg_line_length = round(len(doc.text) / len(lines), 3)
            avg_words_per_line = round(len(words) / len(lines), 3)
            short_line_count = {
                line_threshold: sum([1 for line in lines if len(line) <= line_threshold])
                for line_threshold in self.short_line_max_chars
            }
            short_line_ratio = {
                line_threshold: round(short_line_count / len(lines), 3)
                for line_threshold, short_line_count in short_line_count.items()
            }
            avg_paragraph_length = sum([len(paragraph) for paragraph in paragraphs]) / len(paragraphs)
            short_paragraphs_count = {
                paragraph_threshold: sum([1 for paragraph in paragraphs if len(paragraph.splitlines()) <= paragraph_threshold])
                for paragraph_threshold in self.short_paragraph_max_lines
            }

            short_paragraphs_ratio = {
                paragraph_threshold: short_paragraphs_count / len(paragraphs)
                for paragraph_threshold, short_paragraphs_count in short_paragraphs_count.items()
            }

            for counter, key in zip([default_counter, fqdn_counters, suffix_counters], [None, fqdn, suffix]):
                if key is None:
                    counter["docs_per"]["default"] += 1
                    counter["line_count_per"][len(lines)] += 1
                    counter["avg_line_length_per"][avg_line_length] += 1
                    counter["avg_words_per_line_counter"][avg_words_per_line] += 1
                    counter["paragraphs_num_counter"][len(paragraphs)] +=  1
                    counter["paragraphs_avg_length_counter"][avg_paragraph_length] += 1

                    for short_line_max, count in short_line_count.items():
                        counter[f"short_line_counter_{short_line_max}"][count] += 1

                    for short_line_max, ratio in short_line_ratio.items():
                        counter[f"short_line_ratio_counter_{short_line_max}"][ratio] += 1
                    
                    for short_paragraph_max, count in short_paragraphs_count.items():
                        counter[f"short_paragraphs_counter_{short_paragraph_max}"][count] += 1
                    for short_paragraph_max, ratio in short_paragraphs_ratio.items():
                        counter[f"short_paragraphs_ratio_counter_{short_paragraph_max}"][ratio] += 1
                else:
                    counter["docs_per"][key] += 1
                    counter["line_count_per"][key] += len(lines)
                    counter["avg_line_length_per"][key] += avg_line_length
                    counter["avg_words_per_line_counter"][key] += avg_words_per_line
                    counter["paragraphs_num_counter"][key] += len(paragraphs)
                    counter["paragraphs_avg_length_counter"][key] += avg_paragraph_length
                    for line_threshold, count in short_line_count.items():
                        counter[f"short_line_counter_{line_threshold}"][key] += count
                    for line_threshold, ratio in short_line_ratio.items():
                        counter[f"short_line_ratio_counter_{line_threshold}"][key] += ratio
                    for paragraph_threshold, count in short_paragraphs_count.items():
                        counter[f"short_paragraphs_counter_{paragraph_threshold}"][key] += count
                    for paragraph_threshold, ratio in short_paragraphs_ratio.items():
                        counter[f"short_paragraphs_ratio_counter_{paragraph_threshold}"][key] += ratio

            yield doc

        # save to disk
        with self.output_folder.open(f"{rank:05d}_line_stats.json", "wt") as f:
            json.dump(
            {
                "document": {key: value.to_dict() for key, value in default_counter.items()},
                "fqdn": {key: value.to_dict() for key, value in fqdn_counters.items()},
                "suffix": {key: value.to_dict() for key, value in suffix_counters.items()},
            },
            f
            )