import json
import math
from pathlib import Path

import numpy as np


class AnalysisWriter:
  """Persist numerical results and a readable report to a dedicated directory."""

  def __init__(self, output_dir):
    self.output_dir = Path(output_dir)
    if self.output_dir.exists() and any(self.output_dir.iterdir()):
      raise FileExistsError(f"Analysis output directory is not empty: {self.output_dir}. Choose a new --output directory.")
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def write_table(self, name, frame, compressed=False):
    path = self.output_dir / f"{name}.csv{'.gz' if compressed else ''}"
    frame.to_csv(path, index=False, compression="gzip" if compressed else None)
    return path

  def write_json(self, name, data):
    def clean(value):
      if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
      if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
      if isinstance(value, np.generic):
        return clean(value.item())
      if isinstance(value, float) and not math.isfinite(value):
        return None
      return value

    path = self.output_dir / f"{name}.json"
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path

  def write_report(self, tables, hotelling, metadata):
    sections = ["# OLMS Stability Pilot Analysis", "",
                f"Essays: {metadata['essay_count']}; responses per prompt: {metadata['response_count']}; "
                f"bootstrap replicates: {metadata['bootstrap_replicates']}; seed: {metadata['seed']}.", "",
                "Distances use raw O, L, M, S components and the Euclidean norm. "
                "Self comparisons are bootstrap support only. Essays are held fixed during resampling.", "",
                "Within-prompt bootstrap reconstruction samples responses with replacement, then excludes "
                "same-response ID pairs before computing centroids and distances.", "",
                "Component SDs describe dependent pair scores. Confidence intervals resample responses. "
                "Undefined descriptive statistics are shown as NaN and saved as empty CSV cells.", ""]
    for name, frame in tables.items():
      sections.extend([f"## {name.replace('_', ' ').title()}", "", frame.to_markdown(index=False), ""])
    sections.extend(["## Exploratory Hotelling Test", "", json.dumps(hotelling, indent=2, default=str), "",
                      "Negative component shifts mean lower cross-prompt similarity relative to the same-prompt baseline. "
                      "Overall magnitude is the mean of essay magnitudes, not the magnitude of the mean shift.", "",
                      "Directional bootstrap quantities are fractions of component replicates at or above zero, "
                      "not calibrated confirmatory p-values. Effect estimates and intervals are primary. "
                      "Hotelling's test is exploratory and assumes independent essays and multivariate normality.", "",
                      "No equivalence threshold or cross-essay discriminability test is applied.", ""])
    path = self.output_dir / "report.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path
