# This script generates the semantic similarity aggregate score
# M is for meaning!
import pandas as pd
from TextProcessing.shared_bert_scorer import get_bert_scorer

class M:
  """
  Meaning / Similarity over whole language model responses
  """
  def __init__(self, response_1: str, response_2: str) -> None:
    self.response_1 = response_1
    self.response_2 = response_2
    self.scorer = get_bert_scorer()

  def meaning_similarity(self) -> dict[str, float]:
    """
    Returns the BERTScore between self.response_1 and self.response_2
    """
    if not self.response_1.strip() or not self.response_2.strip():
      raise ValueError("BERTScore meaning similarity requires two non-empty responses.")

    precision, recall, f1 = self.scorer.score([self.response_1], [self.response_2])
    return {
      "precision": precision[0].item(),
      "recall": recall[0].item(),
      "f1": f1[0].item()
    }

  def get_similarity_score(self) -> float:
    m = self.meaning_similarity()
    return m["f1"]


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
  
