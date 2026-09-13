import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from OLMSClasses.OLMS_vector import OLMSVector
from scoring import OLMSPairScorer


class FakeEmbeddingModel:
  def __init__(self, vectors_by_text):
    self.vectors_by_text = vectors_by_text
    self.encode = Mock(side_effect=self._encode)

  def _encode(self, texts):
    return np.asarray([self.vectors_by_text[text] for text in texts], dtype=float)


class FakeBertScorer:
  def score(self, candidates, references):
    return (np.array([0.7]), np.array([0.9]), np.array([0.8]))


class FakeBertScore:
  def __init__(self, clauses_a, clauses_b):
    self.clauses_a = clauses_a
    self.clauses_b = clauses_b

  def create_bert_score_table(self):
    return pd.DataFrame([[(0.7, 0.9, 0.8)]])


class FakeVector:
  instances = []

  def __init__(self, *args, embedding_a=None, embedding_b=None):
    self.embedding_a = embedding_a
    self.embedding_b = embedding_b
    self.structure = Mock()
    self.instances.append(self)

  def create_olms_vector(self):
    return {"org": 0.8, "lex": 0.7, "meaning": 0.9, "struct": 0.6}


class FixedOrganization:
  def __init__(self, **kwargs):
    pass

  def soft_organization_score(self):
    return 0.1


class FixedLexis:
  def __init__(self, **kwargs):
    pass

  def lexical_similarity(self):
    return 0.2


class FixedStructure:
  def __init__(self, **kwargs):
    pass

  def structure_score(self):
    return 0.3


class EmbeddingReuseTests(unittest.TestCase):
  def setUp(self):
    self.rows = [
      {"essay_file": "essay.md", "prompt": "A", "response_index": 1, "response": "A one"},
      {"essay_file": "essay.md", "prompt": "A", "response_index": 2, "response": "A two"},
      {"essay_file": "essay.md", "prompt": "B", "response_index": 1, "response": "B one"},
      {"essay_file": "essay.md", "prompt": "B", "response_index": 2, "response": "B two"},
    ]
    vectors = {
      "A one": [1, 0], "A two": [0, 1], "B one": [1, 1], "B two": [-1, 0],
    }
    self.embedding_model = FakeEmbeddingModel(vectors)
    self.scorer = OLMSPairScorer.__new__(OLMSPairScorer)
    self.scorer.embedding_model = self.embedding_model
    self.scorer.diagnostics_dir = None
    self.scorer.cache = {}
    self.scorer.embedding_cache = {}
    self.scorer.active_essay = None
    self.scorer._prepare = Mock(return_value=(object(), ["Sentence."]))

  def test_prepare_embeddings_encodes_each_response_once_per_essay(self):
    self.scorer.prepare_embeddings([self.rows[0], self.rows[0], *self.rows[1:]])
    self.embedding_model.encode.assert_called_once_with(["A one", "A two", "B one", "B two"])

    self.scorer.prepare_embeddings(self.rows)
    self.embedding_model.encode.assert_called_once()

    next_essay = [{**row, "essay_file": "next.md"} for row in self.rows]
    self.scorer.prepare_embeddings(next_essay)
    self.assertEqual(self.embedding_model.encode.call_count, 2)

  def test_score_pair_uses_cached_embeddings_for_self_and_cross_prompt_pairs(self):
    self.scorer.prepare_embeddings(self.rows)
    FakeVector.instances.clear()
    with patch("TextProcessing.bertscorer.BertScore", FakeBertScore), \
         patch("OLMSClasses.OLMS_vector.OLMSVector", FakeVector):
      self.scorer.score_pair(self.rows[0], self.rows[0])
      self.scorer.score_pair(self.rows[1], self.rows[3])
      self.scorer.score_pair(self.rows[2], self.rows[3])

    self.assertIs(FakeVector.instances[0].embedding_a, FakeVector.instances[0].embedding_b)
    self.assertIs(FakeVector.instances[1].embedding_a, self.scorer.embedding_cache[("essay.md", "A", 2)])
    self.assertIs(FakeVector.instances[1].embedding_b, self.scorer.embedding_cache[("essay.md", "B", 2)])
    self.assertIs(FakeVector.instances[2].embedding_a, self.scorer.embedding_cache[("essay.md", "B", 1)])
    self.assertIs(FakeVector.instances[2].embedding_b, self.scorer.embedding_cache[("essay.md", "B", 2)])
    self.embedding_model.encode.assert_called_once()

  def test_cached_and_uncached_vectors_match(self):
    model = FakeEmbeddingModel({"first": [1, 0], "second": [0, 1]})
    score_table = pd.DataFrame([[(0.7, 0.9, 0.8)]])
    patches = [
      patch("OLMSClasses.OLMS_vector.O", FixedOrganization),
      patch("OLMSClasses.OLMS_vector.L", FixedLexis),
      patch("OLMSClasses.OLMS_vector.S", FixedStructure),
      patch("OLMSClasses.M.get_bert_scorer", return_value=FakeBertScorer()),
    ]
    with patches[0], patches[1], patches[2], patches[3]:
      uncached = OLMSVector(score_table, object(), object(), "first", "second", model)
      cached = OLMSVector(
        score_table, object(), object(), "first", "second", model,
        embedding_a=np.array([1, 0]), embedding_b=np.array([0, 1]),
      )
      self.assertEqual(uncached.create_olms_vector(), cached.create_olms_vector())

    model.encode.assert_called_once_with(["first", "second"])


if __name__ == "__main__":
  unittest.main()
