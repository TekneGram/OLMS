import numpy as np
import pandas as pd


class BootstrapConfidenceInterval:
  """Percentile intervals and the research protocol's directional tail fraction."""

  def __init__(self, confidence=0.95):
    if not 0 < confidence < 1:
      raise ValueError("Confidence must be strictly between zero and one.")
    self.confidence = confidence

  def summarize(self, values, directional=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
      raise ValueError("Bootstrap estimates must be a nonempty finite sample.")
    alpha = (1 - self.confidence) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])

    result = {"confidence": self.confidence, "lower": float(lower),
              "upper": float(upper), "replicates": len(values)}
    if directional:
      result["directional_p_like"] = float(np.mean(values >= 0))
    return result

  def table(self, replicates, group_columns, metrics, directional_metrics=()):
    groups = (replicates.groupby(group_columns, sort=True) 
              if group_columns
              else [((), replicates)])
    rows = []
    for key, group in groups:
      key = key if isinstance(key, tuple) else (key,)
      identity = dict(zip(group_columns, key))
      for metric in metrics:
        rows.append({**identity, "metric": metric,
                      **self.summarize(group[metric], metric in directional_metrics)})
    return pd.DataFrame(rows)
