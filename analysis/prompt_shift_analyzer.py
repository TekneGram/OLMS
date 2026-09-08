import pandas as pd

from analysis.olms_vector_table import OLMSVectorTable
from analysis.vector_centroid_transformer import VectorCentroidTransformer
from analysis.vector_distance_transformer import VectorDistanceTransformer


class PromptShiftAnalyzer:
  """RQ4: cross-prompt similarity minus the average same-prompt baseline."""

  @staticmethod
  def contrast(centroid_a, centroid_b, centroid_ab):
      baseline = (centroid_a + centroid_b) / 2
      delta = centroid_ab - baseline
      return baseline, delta, VectorDistanceTransformer.magnitude(delta)

  def analyze(self, table):
    rows = []
    for essay in table.essays:
      centroids = {
        comparison: VectorCentroidTransformer.transform(
          table.select(essay, comparison)[table.COMPONENTS].to_numpy(dtype=float))
        for comparison in ["AA", "BB", "AB"]
      }
      baseline, delta, magnitude = self.contrast(centroids["AA"], centroids["BB"], centroids["AB"])
      row = {"essay_file": essay, "magnitude": float(magnitude)}
      for i, component in enumerate(table.COMPONENTS):
        for comparison, centroid in centroids.items():
          row[f"{comparison}_{component}"] = centroid[i]
        row[f"within_{component}"] = baseline[i]
        row[f"delta_{component}"] = delta[i]
      rows.append(row)
    per_essay = pd.DataFrame(rows)
    metrics = ["magnitude"] + [f"delta_{c}" for c in OLMSVectorTable.COMPONENTS]
    overall = {"essay_count": len(rows), **per_essay[metrics].mean().to_dict()}
    return per_essay, pd.DataFrame([overall])
