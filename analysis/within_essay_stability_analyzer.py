import pandas as pd

from analysis.descriptive_statistics import DescriptiveStatistics
from analysis.vector_centroid_transformer import VectorCentroidTransformer
from analysis.vector_distance_transformer import VectorDistanceTransformer


class WithinEssayStabilityAnalyzer:
  """RQ1 estimates and individual distances, excluding self comparisons."""

  def analyze(self, table):
    summaries, distances = [], []
    for essay in table.essays:
      for prompt in ["A", "B"]:
        rows = table.select(essay, prompt * 2)
        vectors = rows[table.COMPONENTS].to_numpy(dtype=float)
        centroid = VectorCentroidTransformer.transform(vectors)
        values = VectorDistanceTransformer.transform(vectors, centroid)
        summary = {"essay_file": essay, "prompt": prompt,
                    "response_count": table.response_count, "pair_count": len(rows),
                    "mean_distance": float(values.mean())}
        for component in table.COMPONENTS:
          stats = DescriptiveStatistics.summarize(rows[component])
          summary.update({f"{component}_mean": stats["mean"],
                          f"{component}_sd": stats["sd"]})
        summaries.append(summary)
        rows["distance"] = values
        distances.append(rows)

    return pd.DataFrame(summaries), pd.concat(distances, ignore_index=True)