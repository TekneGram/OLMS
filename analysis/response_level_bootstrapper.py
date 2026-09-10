import numpy as np
import pandas as pd

from analysis.prompt_shift_analyzer import PromptShiftAnalyzer
from analysis.vector_centroid_transformer import VectorCentroidTransformer
from analysis.vector_distance_transformer import VectorDistanceTransformer


class ResponseLevelBootstrapper:
  """Resample response IDs; cached pair scores retain duplicate multiplicities."""

  def __init__(self, replicates=5000, seed=42):
    if not isinstance(replicates, int) or replicates < 1:
      raise ValueError("Bootstrap replicates must be a positive integer.")
    if not isinstance(seed, int) or seed < 0:
      raise ValueError("Bootstrap seed must be a nonnegative integer.")
    self.replicates = replicates
    self.seed = seed

  def _draw_samples(self, rng, count, size):
    samples = rng.integers(0, count, size=(size, count))
    for row in range(size):
      while len(np.unique(samples[row])) < 2:
        samples[row] = rng.integers(0, count, size=count)
    return samples

  def _within_prompt_bootstrap(self, matrix, samples, left, right):
    centroids = np.empty((samples.shape[0], 4), dtype=float)
    mean_distances = np.empty(samples.shape[0], dtype=float)

    for row, sample in enumerate(samples):
      source_ids = sample[left]
      target_ids = sample[right]
      keep = source_ids != target_ids
      if not keep.any():
        raise ValueError("Bootstrap sample must contain at least two distinct responses.")

      pairs = matrix[source_ids[keep], target_ids[keep]]
      centroid = VectorCentroidTransformer.transform(pairs)
      centroids[row] = centroid
      mean_distances[row] = VectorDistanceTransformer.transform(pairs, centroid).mean()

    return centroids, mean_distances

  def run(self, table):
    rng = np.random.default_rng(self.seed)
    within, shifts = [], []
    count = table.response_count
    left, right = np.triu_indices(count, k=1)
    # Bound temporary pair arrays even when the requested replicate count is large.
    for essay in table.essays:
      matrices = table.matrices(essay)
      for start in range(0, self.replicates, 250):
        size = min(250, self.replicates - start)
        samples = {p: self._draw_samples(rng, count, size) for p in ["A", "B"]}
        centroids = {}
        replicate_ids = np.arange(start + 1, start + size + 1)

        for prompt in ["A", "B"]:
          sample = samples[prompt]
          centroid, distance = self._within_prompt_bootstrap(
            matrices[prompt * 2], sample, left, right)
          centroids[prompt] = centroid
          data = {"essay_file": essay, "prompt": prompt,
                  "replicate": replicate_ids, "mean_distance": distance}

          for i, component in enumerate(table.COMPONENTS):
            data[f"{component}_mean"] = centroid[:, i]
          within.append(pd.DataFrame(data))

        ab = matrices["AB"][samples["A"][:, :, None], samples["B"][:, None, :]]
        centroid_ab = VectorCentroidTransformer.transform(ab.reshape(size, count * count, 4))
        _, delta, magnitude = PromptShiftAnalyzer.contrast(centroids["A"], centroids["B"], centroid_ab)
        data = {"essay_file": essay, "replicate": replicate_ids, "magnitude": magnitude}

        for i, component in enumerate(table.COMPONENTS):
          data[f"delta_{component}"] = delta[:, i]
        shifts.append(pd.DataFrame(data))

    shift_frame = pd.concat(shifts, ignore_index=True)
    metrics = ["magnitude"] + [f"delta_{c}" for c in table.COMPONENTS]
    overall = shift_frame.groupby("replicate", as_index=False)[metrics].mean()

    return {
        "within_bootstrap": pd.concat(within, ignore_index=True),
        "prompt_shift_bootstrap": shift_frame,
        "overall_prompt_shift_bootstrap": overall,
    }
