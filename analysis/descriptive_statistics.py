import numpy as np


class DescriptiveStatistics:
  """Univariate descriptive summaries; undefined statistics are NaN."""

  @staticmethod
  def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
      raise ValueError("Statistics require a nonempty finite one-dimensional sample.")

    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    
    return {
      "count": int(values.size), "mean": mean, "sd": sd,
      "min": float(values.min()), "max": float(values.max()),
      "q1": float(q1), "median": float(median), "q3": float(q3),
      "iqr": float(q3 - q1), "range": float(np.ptp(values)),
      "cv": sd / mean if mean != 0 else float("nan"),
    }
