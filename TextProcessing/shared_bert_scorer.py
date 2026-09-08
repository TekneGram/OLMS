from functools import lru_cache

from bert_score import BERTScorer
from transformers import logging

logging.set_verbosity_error()


@lru_cache(maxsize=1)
def get_bert_scorer() -> BERTScorer:
  return BERTScorer(lang="en", rescale_with_baseline=True)
