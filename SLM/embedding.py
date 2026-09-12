"""Shared GGUF embedding-model support for OLMS scoring."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import configuration
from llama_cpp import Llama


class EmbeddingModel:
  """Load a GGUF embedding model once and encode batches of text."""

  def __init__(self) -> None:
    self._model: Llama | None = Llama(
      model_path=configuration.embeddings_model,
      n_ctx=configuration.n_ctx,
      n_threads=configuration.n_threads,
      n_gpu_layers=configuration.n_gpu_layers,
      verbose=configuration.verbose,
      embedding=True,
    )

  def encode(self, texts: Sequence[str]) -> np.ndarray:
    """Return one embedding vector for each supplied text."""
    if self._model is None:
      raise RuntimeError("The embedding model has already been closed.")

    response = self._model.create_embedding(list(texts))
    vectors = np.asarray(
      [item["embedding"] for item in response["data"]], dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(texts):
      raise ValueError("Embedding model returned an unexpected number of vectors.")
    return vectors

  def close(self) -> None:
    """Release the native llama.cpp model resources."""
    if self._model is not None:
      self._model.close()
      self._model = None

  def __enter__(self) -> EmbeddingModel:
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()
