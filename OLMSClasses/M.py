# This script generates the semantic similarity aggregate score
# M is for meaning!
import pandas as pd

class M:
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
  