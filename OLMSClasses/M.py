# This script generates the semantic similarity aggregate score
# M is for meaning!
import pandas as pd
import numpy as np
from collections.abc import Sequence
from SLM.embedding import EmbeddingModel
from TextProcessing.shared_bert_scorer import get_bert_scorer

class M:
  """
  Meaning / Similarity over whole language model responses
  """
  def __init__(
      self,
      response_1: str,
      response_2: str,
      embedding_model: EmbeddingModel,
      embedding_1: np.ndarray | None = None,
      embedding_2: np.ndarray | None = None,
      bertscore: dict[str, float] | None = None,
  ) -> None:
    self.response_1 = response_1
    self.response_2 = response_2
    self.scorer = get_bert_scorer()
    self.embedding_model = embedding_model
    self.embedding_1 = embedding_1
    self.embedding_2 = embedding_2
    self.bertscore = bertscore

  @staticmethod
  def batch_meaning_similarity_bertscores(
      response_pairs: Sequence[tuple[str, str]],
      scorer=None,
      context_cache=None,
  ) -> list[dict[str, float]]:
    """Return one document-level BERTScore result for each complete response pair."""
    if not response_pairs:
      return []
    if any(not first.strip() or not second.strip() for first, second in response_pairs):
      raise ValueError("BERTScore meaning similarity requires two non-empty responses.")

    if scorer is None and context_cache is None:
      scorer = get_bert_scorer()
    score_source = context_cache if context_cache is not None else scorer
    precision, recall, f1 = score_source.score(
      [first for first, _ in response_pairs],
      [second for _, second in response_pairs],
    )
    if not (len(precision) == len(recall) == len(f1) == len(response_pairs)):
      raise ValueError("BERTScore returned an unexpected number of document scores.")
    return [
      {"precision": precision[index].item(), "recall": recall[index].item(), "f1": f1[index].item()}
      for index in range(len(response_pairs))
    ]

  def meaning_similarity_bertscore(self) -> dict[str, float]:
    """
    Returns the BERTScore between self.response_1 and self.response_2
    """
    if self.bertscore is not None:
      return self.bertscore
    return self.batch_meaning_similarity_bertscores(
      [(self.response_1, self.response_2)], self.scorer)[0]

  def meaning_similarity_embeddings(self) -> float:
    """Return transformed cosine similarity between the response embeddings."""
    if not self.response_1.strip() or not self.response_2.strip():
      raise ValueError("Embedding meaning similarity requires two non-empty responses.")

    if self.embedding_1 is not None and self.embedding_2 is not None:
      first, second = self.embedding_1, self.embedding_2
    else:
      embeddings = self.embedding_model.encode([self.response_1, self.response_2])
      first, second = embeddings
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
      raise ValueError("Embedding meaning similarity requires non-zero embedding vectors.")

    cosine_score = float(np.dot(first, second) / denominator)
    embedding_score = (cosine_score + 1.0) / 2.0
    return max(0.0, min(1.0, embedding_score))

  def get_similarity_score(self) -> float:
    bertscore = self.meaning_similarity_bertscore()
    embedding_score = self.meaning_similarity_embeddings()
    return (bertscore["f1"] + embedding_score) / 2.0


class M_old:
  """
  Meaning / Semantic similarity over matched clause pairs.
  Implements:

    M = (1 / |R|) * sum_{(i, j) in R} BERTScoreF1(A_i, B_j)
  where R is the set of matched clause pairs returned by PairMatcher.match().
  """

  REQUIRED_COLUMNS = {
    "clause_in_a_index",
    "clause_in_b_index",
    "f1"
  }

  def __init__(
      self,
      match_table: pd.DataFrame
  ) -> None:
    missing_columns = self.REQUIRED_COLUMNS - set(match_table.columns)
    if missing_columns:
      raise ValueError(
        f"The paired clauses table from pair_matcher.py is missing required columns: {sorted(missing_columns)}"
      )

    self.match_table = match_table.copy()
    return

  def semantics_score(self) -> float:
    """
    Return the average BERTScore F1 over matched clause pairs.
    If there are no matched pairs, return 0.0
    """
    if self.match_table.empty:
      return 0.0
    
    return float(self.match_table["f1"].mean())
  
