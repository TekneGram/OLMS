import unittest
from unittest.mock import Mock, patch

import numpy as np

from OLMSClasses.M import M
from SLM.embedding import EmbeddingModel


class FakeEmbeddingModel:
  def __init__(self, vectors):
    self.vectors = np.asarray(vectors, dtype=float)
    self.encode = Mock(return_value=self.vectors)


class FakeBertScorer:
  def __init__(self, f1=0.8):
    self.f1 = f1

  def score(self, candidates, references):
    return (np.array([0.7]), np.array([0.9]), np.array([self.f1]))


class MeaningTests(unittest.TestCase):
  def make_meaning(self, vectors, f1=0.8):
    scorer = FakeBertScorer(f1)
    model = FakeEmbeddingModel(vectors)
    with patch("OLMSClasses.M.get_bert_scorer", return_value=scorer):
      meaning = M("First response", "Second response", model)
    return meaning, model

  def test_bertscore_method_is_renamed_and_preserves_scores(self):
    meaning, _ = self.make_meaning([[1, 0], [0, 1]])
    self.assertEqual(
      meaning.meaning_similarity_bertscore(),
      {"precision": 0.7, "recall": 0.9, "f1": 0.8})
    self.assertFalse(hasattr(meaning, "meaning_similarity"))

  def test_embedding_similarity_uses_cosine_and_affine_transform(self):
    for vectors, expected in [
        ([[1, 0], [1, 0]], 1.0),
        ([[1, 0], [0, 1]], 0.5),
        ([[1, 0], [-1, 0]], 0.0),
    ]:
      with self.subTest(vectors=vectors):
        meaning, model = self.make_meaning(vectors)
        self.assertAlmostEqual(meaning.meaning_similarity_embeddings(), expected)
        model.encode.assert_called_once_with(["First response", "Second response"])

  def test_total_meaning_score_averages_bertscore_f1_and_embedding_score(self):
    meaning, _ = self.make_meaning([[1, 0], [0, 1]], f1=0.8)
    self.assertAlmostEqual(meaning.get_similarity_score(), 0.65)

  def test_embedding_similarity_rejects_zero_vectors_and_empty_responses(self):
    meaning, _ = self.make_meaning([[0, 0], [1, 0]])
    with self.assertRaisesRegex(ValueError, "non-zero"):
      meaning.meaning_similarity_embeddings()

    model = FakeEmbeddingModel([[1, 0], [1, 0]])
    with patch("OLMSClasses.M.get_bert_scorer", return_value=FakeBertScorer()):
      empty = M(" ", "Second response", model)
    with self.assertRaisesRegex(ValueError, "BERTScore"):
      empty.meaning_similarity_bertscore()
    with self.assertRaisesRegex(ValueError, "Embedding"):
      empty.meaning_similarity_embeddings()

  def test_embedding_model_encodes_and_closes_native_model(self):
    native = Mock()
    native.create_embedding.return_value = {
      "data": [
        {"embedding": [1.0, 2.0]},
        {"embedding": [3.0, 4.0]},
      ]
    }
    with patch("SLM.embedding.Llama", return_value=native) as llama:
      model = EmbeddingModel()
      np.testing.assert_array_equal(
        model.encode(["first", "second"]), np.array([[1.0, 2.0], [3.0, 4.0]]))
      model.close()
      model.close()

    self.assertTrue(llama.call_args.kwargs["embedding"])
    native.create_embedding.assert_called_once_with(["first", "second"])
    native.close.assert_called_once_with()
    with self.assertRaisesRegex(RuntimeError, "already been closed"):
      model.encode(["later"])


if __name__ == "__main__":
  unittest.main()
