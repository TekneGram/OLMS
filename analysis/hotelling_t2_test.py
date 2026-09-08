import numpy as np
from scipy.stats import f


class HotellingT2Test:
  """Exploratory one-sample test on the four essay-level prompt shifts."""

  def run(self, shifts):
    values = np.asarray(shifts, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
      raise ValueError("Hotelling's test requires finite essay-level four-component shifts.")

    count, dimensions = values.shape
    result = {"exploratory": True, "essay_count": count, "dimensions": dimensions,
              "df1": dimensions, "df2": count - dimensions,
              "t2": None, "f": None, "p_value": None}

    if count <= dimensions:
      return {**result, "status": "unavailable", "reason": "Requires more than four essays."}

    covariance = np.cov(values, rowvar=False, ddof=1)
    condition = float(np.linalg.cond(covariance))
    result["covariance_condition_number"] = condition
    mean = values.mean(axis=0)

    # Compare centered variation against the precision of the original shifts,
    # so roundoff in otherwise constant shifts cannot masquerade as full rank.
    tolerance = np.finfo(float).eps * max(values.shape) * np.linalg.norm(values, ord=2)
    rank = np.linalg.matrix_rank(values - mean, tol=tolerance)
    if rank < dimensions or not np.isfinite(condition) or condition > 1e12:
      return {**result, "status": "unavailable", "reason": "Covariance is singular or numerically ill-conditioned."}

    t2 = float(count * mean @ np.linalg.solve(covariance, mean))
    statistic = (count - dimensions) / (dimensions * (count - 1)) * t2

    return {**result, "status": "ok", "t2": t2, "f": statistic,
            "p_value": float(f.sf(statistic, dimensions, count - dimensions))}
