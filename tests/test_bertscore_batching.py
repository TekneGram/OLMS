import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from TextProcessing.bertscorer import BertScore
from scoring import OLMSPairScorer
import configuration


class DeterministicBertScorer:
  def __init__(self):
    self.calls = []

  def score(self, candidates, references):
    self.calls.append((candidates, references))
    values = np.array([
      sum(map(ord, candidate + "|" + reference)) / 1000.0
      for candidate, reference in zip(candidates, references)
    ])
    return values, values + 1.0, values + 2.0


class ShortBertScorer:
  def score(self, candidates, references):
    size = len(candidates) - 1
    return np.zeros(size), np.zeros(size), np.zeros(size)


class BertScoreBatchingTests(unittest.TestCase):
  def test_reconstructs_complete_uneven_matrices_without_cross_pair_cells(self):
    first = (["A one", "A two"], ["B one", "B two", "B three"])
    second = (["C one"], ["D one", "D two"])
    scorer = DeterministicBertScorer()

    tables = BertScore.create_bert_score_tables([first, second], scorer)

    self.assertEqual([table.shape for table in tables], [(2, 3), (1, 2)])
    self.assertEqual(len(scorer.calls), 1)
    candidates, references = scorer.calls[0]
    self.assertEqual(len(candidates), 8)
    self.assertEqual(candidates[6:], ["C one", "C one"])
    self.assertEqual(references[6:], ["D one", "D two"])

    with patch("TextProcessing.bertscorer.get_bert_scorer", return_value=scorer):
      single_first = BertScore(*first).create_bert_score_table()
      single_second = BertScore(*second).create_bert_score_table()
    pd.testing.assert_frame_equal(tables[0], single_first)
    pd.testing.assert_frame_equal(tables[1], single_second)

  def test_rejects_a_result_that_cannot_fill_every_complete_matrix(self):
    with self.assertRaisesRegex(ValueError, "unexpected number"):
      BertScore.create_bert_score_tables([(["A"], ["B", "C"])], ShortBertScorer())

  def test_oom_backoff_halves_one_group_then_restores_the_configured_size(self):
    scorer = OLMSPairScorer.__new__(OLMSPairScorer)
    calls = []

    def score_batch(pairs):
      calls.append(len(pairs))
      if len(calls) == 1:
        raise RuntimeError("MPS backend out of memory")
      return [{"pair": pair[0]} for pair in pairs]

    scorer._score_pair_batch = Mock(side_effect=score_batch)
    scorer.prepare_bertscore_embeddings = Mock()
    pairs = [(f"pair-{index}", f"other-{index}") for index in range(8)]
    with patch.object(configuration, "bertscore_pair_batch_size", 4), \
         patch("scoring.clear_bertscore_memory") as clear_memory:
      result = list(scorer.score_pairs(pairs))

    self.assertEqual(calls, [4, 2, 2, 4])
    self.assertEqual(result, [{"pair": f"pair-{index}"} for index in range(8)])
    clear_memory.assert_called_once_with()

  def test_backoff_reraises_non_memory_errors_and_single_pair_oom(self):
    scorer = OLMSPairScorer.__new__(OLMSPairScorer)
    scorer._score_pair_batch = Mock(side_effect=RuntimeError("invalid BERTScore inputs"))
    scorer.prepare_bertscore_embeddings = Mock()
    with patch.object(configuration, "bertscore_pair_batch_size", 4), \
         patch("scoring.clear_bertscore_memory") as clear_memory:
      with self.assertRaisesRegex(RuntimeError, "invalid BERTScore"):
        list(scorer.score_pairs([("first", "second")]))
      clear_memory.assert_not_called()

    scorer._score_pair_batch = Mock(side_effect=RuntimeError("out of memory"))
    with patch.object(configuration, "bertscore_pair_batch_size", 4):
      with self.assertRaisesRegex(RuntimeError, "out of memory"):
        list(scorer.score_pairs([("first", "second")]))


if __name__ == "__main__":
  unittest.main()
