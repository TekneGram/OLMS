import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from TextProcessing.bertscorer import BertScore


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


if __name__ == "__main__":
  unittest.main()
