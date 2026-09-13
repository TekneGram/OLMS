from functools import lru_cache

from bert_score import BERTScorer
import torch
from transformers import logging

import configuration

logging.set_verbosity_error()


def _mps_is_available() -> bool:
  mps = getattr(torch.backends, "mps", None)
  return bool(mps is not None and mps.is_built() and mps.is_available())


def resolve_bertscore_device() -> str:
  """Resolve the configured BERTScore device, preferring Apple MPS in auto mode."""
  requested = configuration.bertscore_device
  if requested == "auto":
    return "mps" if _mps_is_available() else "cpu"
  if requested == "cpu":
    return "cpu"
  if requested == "mps":
    if not _mps_is_available():
      raise RuntimeError("BERTScore is configured for MPS, but MPS is unavailable.")
    return "mps"
  raise ValueError("bertscore_device must be 'auto', 'cpu', or 'mps'.")


@lru_cache(maxsize=1)
def get_bert_scorer() -> BERTScorer:
  return BERTScorer(
    lang="en",
    rescale_with_baseline=True,
    device=resolve_bertscore_device(),
  )
