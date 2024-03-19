
import argparse
from collections import defaultdict
import json
import os.path

from loguru import logger
from tqdm import tqdm

from datatrove.io import get_datafolder, open_file
from datatrove.utils.stats import MetricStats, MetricStatsDict, PipelineStats, Stats
import matplotlib.pyplot as plt

import json
import os
import argparse

def load_and_merge_stats(stats_folder):
    stats = defaultdict(lambda: defaultdict(list))

    for file_name in tqdm(stats_folder.list_files(), desc="Loading and processing stats files"):
        with stats_folder.open(file_name, "rt") as file:
            stats_data = json.load(file)
            for group_name, group_metrics in stats_data.items():
                for metric_name, metric_data in group_metrics.items():
                    stats[group_name][metric_name].append(MetricStatsDict(init=metric_data))

    merged = {
        group: {
            metric_name: sum(metric_stats, start=MetricStatsDict())
            for metric_name, metric_stats in metrics.items()
        }
        for group, metrics in stats.items()
    }
    logger.info(f"Merging completed.")
    return merged

def prepare_data(stats, group_by, top_k=100):
    histograms = {}
    grouped_data = stats[group_by]
    norm = grouped_data["docs_per"]
    for metric, values  in grouped_data.items():
        if metric == "docs_per":
            continue

        values_total = {k: v.total for k, v in values.items()}
        sorted_items = sorted(values_total.items(), key=lambda item: item[1], reverse=True)[:top_k]
        histogram = {k: round(v / norm["default" if group_by == "document" else k].total, 3) for k, v in sorted_items}
        histograms[metric] = histogram
    return histograms


def plot_histograms(histograms: list[dict[str, float]]):
    for stat_name, histogram in histograms.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        if all(isinstance(k, str) for k in histogram.keys()):
            x = [k for k, v in sorted(histogram.items(), key=lambda item: item[1])]
        else:
            x = sorted(histogram.keys())

        y = [histogram[k] for k in x]

        ax.plot(x, y)
        ax.set_title(f"Line Plot for {stat_name}")
        ax.set_xlabel(stat_name)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process word stats summaries.")
    parser.add_argument("folder", type=str, help="Folder containing summary results.")
    parser.add_argument("--group_by", type=str, help="Group by this key.", default="document")
    parser.add_argument("--top_k", type=int, help="Consider top k keys.", default=100)
    
    args = parser.parse_args()
    
    stats = load_and_merge_stats(get_datafolder(args.folder))
    data = prepare_data(stats, args.group_by, args.top_k)
    

    plot_histograms(data)





if __name__ == "__main__":
    main()