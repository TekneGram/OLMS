import pandas as pd

from analysis.descriptive_statistics import DescriptiveStatistics
from analysis.olms_vector_table import OLMSVectorTable


class BetweenEssayStabilityAnalyzer:
  """RQ2: describe essay-level estimates separately for each prompt."""

  def analyze(self, within_estimates):
    metrics = ["mean_distance"] + [f"{c}_mean" for c in OLMSVectorTable.COMPONENTS]
    rows = []
    for prompt, group in within_estimates.groupby("prompt", sort=True):
      for metric in metrics:
        rows.append({"prompt": prompt, "metric": metric,
                      **DescriptiveStatistics.summarize(group[metric])})
    return pd.DataFrame(rows)
