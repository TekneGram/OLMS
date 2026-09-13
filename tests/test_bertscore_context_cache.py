import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from OLMSClasses.M import M
from TextProcessing.bertscore_context_cache import BertScoreContextCache
from TextProcessing.bertscorer import BertScore


class FakeTokenizer:
  sep_token_id = 102
  cls_token_id = 101


class FakeModel:
  def __init__(self):
    self.parameter = torch.nn.Parameter(torch.zeros(1))

  def parameters(self):
    return iter([self.parameter])


class FakeScorer:
  def __init__(self):
    self._model = FakeModel()
    self._tokenizer = FakeTokenizer()
    self.device = "cpu"
    self.all_layers = False
    self.idf = False
    self._idf_dict = None
    self.rescale_with_baseline = False


class RecordingScoreSource:
  def __init__(self):
    self.calls = []

  def score(self, candidates, references):
    self.calls.append((list(candidates), list(references)))
    values = np.arange(len(candidates), dtype=float) / 10.0
    return values, values + 0.1, values + 0.2


class BertScoreContextCacheTests(unittest.TestCase):
  def test_encodes_each_distinct_text_once_and_keeps_cached_tensors_immutable(self):
    calls = []

    def fake_get_bert_embedding(texts, *args, **kwargs):
      calls.append(list(texts))
      lengths = [len(text.split()) + 2 for text in texts]
      maximum = max(lengths)
      embeddings = torch.zeros(len(texts), maximum, 2)
      masks = torch.zeros(len(texts), maximum, dtype=torch.long)
      idfs = torch.zeros(len(texts), maximum)
      for row, (text, length) in enumerate(zip(texts, lengths, strict=True)):
        base = sum(map(ord, text))
        for token in range(length):
          embeddings[row, token] = torch.tensor([base + token + 1.0, token + 2.0])
        masks[row, :length] = 1
        idfs[row, :length] = 1
      return embeddings, masks, idfs

    cache = BertScoreContextCache(FakeScorer())
    with patch("TextProcessing.bertscore_context_cache.get_bert_embedding", fake_get_bert_embedding):
      cache.prepare(["same text", "other text", "same text"])
      stored = {text: tuple(value.clone() for value in stats) for text, stats in cache._stats.items()}
      first = cache.score(["same text", "other text"], ["other text", "same text"])
      second = cache.score(["same text"], ["other text"])

    self.assertEqual(sum(len(batch) for batch in calls), 2)
    self.assertEqual(set(cache._stats), {"same text", "other text"})
    for text, (embedding, idf) in stored.items():
      torch.testing.assert_close(cache._stats[text][0], embedding)
      torch.testing.assert_close(cache._stats[text][1], idf)
    self.assertEqual(tuple(value.shape for value in first), ((2,), (2,), (2,)))
    self.assertEqual(tuple(value.shape for value in second), ((1,), (1,), (1,)))

  def test_sentence_and_document_paths_use_the_supplied_cached_score_source(self):
    source = RecordingScoreSource()
    tables = BertScore.create_bert_score_tables(
      [(["A one", "A two"], ["B one", "B two"])], context_cache=source)
    meanings = M.batch_meaning_similarity_bertscores(
      [("Whole A", "Whole B")], context_cache=source)

    self.assertEqual(len(source.calls), 2)
    self.assertEqual(source.calls[0], (
      ["A one", "A one", "A two", "A two"],
      ["B one", "B two", "B one", "B two"],
    ))
    self.assertEqual(tables[0].shape, (2, 2))
    self.assertEqual(meanings, [{"precision": 0.0, "recall": 0.1, "f1": 0.2}])

  @unittest.skipUnless(
    os.environ.get("OLMS_RUN_BERTSCORE_INTEGRATION") == "1",
    "set OLMS_RUN_BERTSCORE_INTEGRATION=1 to run against the installed BERTScore model",
  )
  def test_cached_scores_match_installed_bertscore(self):
    from TextProcessing.shared_bert_scorer import get_bert_scorer

    scorer = get_bert_scorer()
    candidates = ["The student revised the argument.", "The conclusion is concise."]
    references = ["The learner improved the argument.", "The ending is brief."]
    uncached = scorer.score(candidates, references)
    cached = BertScoreContextCache(scorer).score(candidates, references)
    for actual, expected in zip(cached, uncached, strict=True):
      torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
