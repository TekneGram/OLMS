import numpy as np


class VectorCentroidTransformer:
  """Calculate centroids along the pair axis of OLMS arrays."""

  @staticmethod
  def transform(vectors):
    values = np.asarray(vectors, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 4 or values.shape[-2] == 0:
      raise ValueError("Expected nonempty arrays of four-component OLMS vectors.")
    if not np.isfinite(values).all():
      raise ValueError("OLMS vectors must contain finite values.")
    return values.mean(axis=-2)
