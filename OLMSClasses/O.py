# This script contains classes that calculate the Organization part of the OLMS vector.
# Kendall's Tau variant b is used here so that it is tie-aware (because clause matches can be many to many)
from scipy.stats import kendalltau
import pandas as pd
import math

class O:
  """
  Organization measures for matched clauses.

  This class calculates the Organization components inspired by the coherence section of Jang et al. (2025)

  Note: this class is named O for Organization. It is not the same as Jang et al.'s (2025) O in A.1.3, p. 5701.
  """

  REQUIRED_COLUMNS = {
    "clause_in_a_index",
    "clause_in_b_index"
  }

  def __init__(
      self,
      match_table: pd.DataFrame,
      n_clauses_a: int,
      n_clauses_b: int,
  ) -> None:
    # Validate the match_table
    missing_columns = self.REQUIRED_COLUMNS - set(match_table.columns)
    if missing_columns:
      raise ValueError(
        f"The paired clauses table from pair_matcher.py is missing required columns: {sorted(missing_columns)}"
      )

    self.match_table = match_table.copy()
    self.n_clauses_a = n_clauses_a
    self.n_clauses_b = n_clauses_b

  def normalized_tie_aware_tau(self) -> float:
    """
    Calculate hte normalized order-correlation component
      O = (tau + 1) / 2
    using Kendall's Tau-b, which is tie-aware

    Returns a value in [0, 1]:
      1.0 means matched clauses preserve the same order.
      0.5 means no clear order relationship / neutral
      0.0 means perfectly reversed order
    """
    if self.match_table.empty:
      return 0.0

    if len(self.match_table) == 1:
      return 1.0

    clause_indices_a = self.match_table["clause_in_a_index"]
    clause_indices_b = self.match_table["clause_in_b_index"]

    result = kendalltau(
      clause_indices_a,
      clause_indices_b,
      variant="b"
    )

    tau = result.statistic

    if math.isnan(tau):
      return 0.5

    return (tau + 1.0) / 2.0

  def position_matching(self) -> float:
    """
    Calculate the average normalized position similarity across matched clauses.

    For each matched pair (i, j):

      p_i = i / (m_a - 1)
      p_j = j / (m_b - 1)
      P_ij = 1 - abs(p_i - p_j)

    Returns a value in [0, 1]
    """

    if self.match_table.empty:
      return 0.0

    denominator_a = self.n_clauses_a - 1
    denominator_b = self.n_clauses_b - 1

    if denominator_a <= 0:
      normalized_a = 0.0
    else:
      normalized_a = (
        self.match_table["clause_in_a_index"] / denominator_a
      )

    if denominator_b <= 0:
      normalized_b = 0.0
    else:
      normalized_b = (
        self.match_table["clause_in_b_index"] / denominator_b
      )

    pair_scores = 1.0 - (normalized_a - normalized_b).abs()
    
    return float(pair_scores.mean())

  def sequential_matching(self) -> float:
    if self.match_table.empty:
      return 0.0

    a_to_b = (
      self.match_table
      .groupby("clause_in_a_index")["clause_in_b_index"]
      .apply(set)
      .to_dict()
    )

    b_to_a = (
      self.match_table
      .groupby("clause_in_b_index")["clause_in_a_index"]
      .apply(set)
      .to_dict()
    )

    score_a = self._directional_continuity(
      source_to_target = a_to_b,
      n_source_clauses = self.n_clauses_a,
    )

    score_b = self._directional_continuity(
      source_to_target = b_to_a,
      n_source_clauses = self.n_clauses_b,
    )

    return float((score_a + score_b) / 2.0)

  def _directional_continuity(
      self,
      source_to_target: dict[int, set[int]],
      n_source_clauses: int,
  ) -> float:
    if n_source_clauses <= 1:
      return 1.0

    preserved = 0
    comparable = 0

    for source_index in range(n_source_clauses - 1):
      current_targets = source_to_target.get(source_index)
      next_targets = source_to_target.get(source_index + 1)

      if not current_targets or not next_targets:
        continue

      comparable += 1

      transition_is_preserved = any(
        next_target == current_target + 1
        for current_target in current_targets
        for next_target in next_targets
      )

      if transition_is_preserved:
        preserved += 1

    if comparable == 0:
      return 0.0

    return preserved / comparable

  def organization_score(self) -> float:
    norm_tau = self.normalized_tie_aware_tau()
    position_score = self.position_matching()
    sequential_score = self.sequential_matching()
    return (0.34*norm_tau + 0.33*position_score + 0.33*sequential_score)