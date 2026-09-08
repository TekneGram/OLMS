import numpy as np


class VectorDistanceTransformer:
  """Euclidean distances and magnitudes, without component rescaling."""

  @staticmethod
  def magnitude(vectors):
    values = np.asarray(vectors, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4 or not np.isfinite(values).all():
      raise ValueError("Expected finite four-component OLMS vectors.")
    return np.linalg.norm(values, axis=-1)

  @classmethod
  def transform(cls, vectors, centroid):
    return cls.magnitude(np.asarray(vectors) - np.expand_dims(centroid, axis=-2))
