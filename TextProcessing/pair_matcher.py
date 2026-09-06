import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

class PairMatcher:
  """
  Hybrid clause matcher
  
  1. Use Hungarian matching to find the best one-to-one clause matches
  2. For clauses left unmatched after Hungarian matching, greedily assign them to their best remaining eligible partner.
  3. No clause may participate in more than `capacity` matches
  4. Pairs below `min_score` are never matched

  Parameters
  ----------
  score_table:
    DataFrame returned by BertScore.create_bert_score_table()
    Each cell must contains [precision, recall, f1]

  min_score:
    Minimum score required for a pair to be eligible

  metric:
    One of `precision`, `recall`, `f1`

  capacity
    Maximum number of matches allowed for any clause
    Default is 2
  """

  METRIC_INDEX = {
    "precision": 0,
    "recall": 1,
    "f1": 2
  }

  def __init__(
      self,
      score_table: pd.DataFrame,
      min_score: float = 0.0,
      metric: str = "f1",
      capacity: int = 2,
  ) -> None:
    if metric not in self.METRIC_INDEX:
      raise ValueError(
        "metric must be `precision`, `recall` or `f1`"
      )

    if capacity < 1:
      raise ValueError("capacity must be at least 1.")

    self.score_table = score_table
    self.min_score = min_score
    self.metric = metric
    self.capacity = capacity

    self.n_a = len(score_table.index)   # Number of clauses in A
    self.n_b = len(score_table.columns) # Number of clauses in B

    self._scores = self._extract_scores()

  def _extract_scores(self) -> np.ndarray:
    """
    Extract the selected BERTScore component into a numeric matrix
    """

    metric_index = self.METRIC_INDEX[self.metric]

    scores = np.zeros(
      (self.n_a, self.n_b),
      dtype=float
    )

    for i in range(self.n_a):
      for j in range(self.n_b):
        cell = self.score_table.iat[i, j]
        scores[i, j] = float(cell[metric_index])

    return scores

  def _hungarian_core(self) -> list[tuple[int, int]]:
    """
    Find the best one-to-one core alignment between clauses
    using Hungarian matching
    Only pairs meeting min_score are accepted
    """

    # scipy.optimize.linear_sum_assignment minimizes cost
    # Negating the similarity scores turns maximization into minimization
    cost_matrix = -self._scores.copy()

    # Give ineligible pairs a very large cost to Hungarian matching avoids
    # these wherever possible
    ineligible = self._scores < self.min_score
    cost_matrix[ineligible] = 1_000_000.0

    row_indices, col_indices = linear_sum_assignment(
      cost_matrix
    )

    matches = []

    for i, j in zip(row_indices, col_indices):
      # Hungarian may still assign an ineligible pair if there are no other possibilities
      # so explicitly reject it
      if self._scores[i, j] < self.min_score:
        continue

      matches.append((i, j))
    return matches

  def _add_leftover_matches(
      self,
      matches: list[tuple[int, int]],
  ) -> list[tuple[int, int]]:
    """
    Greedily matches clauses left unmatched by the Hungarian core to their
    highest-scoring eligible partner available at that moment, subject to capacity
    """

    selected = set(matches)

    # Count how many matches each clause already has
    count_a = np.zeros(self.n_a, dtype=int)
    count_b = np.zeros(self.n_b, dtype=int)

    for i, j in matches:
      count_a[i] += 1
      count_b[j] += 1

    candidates = []

    for i in range(self.n_a):
      for j in range(self.n_b):

        if count_a[i] > 0 and count_b[j] > 0:
          continue

        if self._scores[i, j] < self.min_score:
          continue

        if (i, j) in selected:
          continue

        candidates.append(
          (
            self._scores[i, j],
            i,
            j,
          )
        )

    candidates.sort(
      key=lambda candidate: (
        -candidate[0],
        candidate[1],
        candidate[2]
      )
    )

    for _, i, j in candidates:

      if count_a[i] >= self.capacity:
        continue

      if count_b[j] >= self.capacity:
        continue

      selected.add((i, j))
      count_a[i] += 1
      count_b[j] += 1

    return list(selected)

  def match(self) -> pd.DataFrame:
    """
    Perform hybrid Hungarian + leftover matching

    Returns
    ---------
    pd.DataFrame
        One row per selected clause match
    """

    # Step 1: globally optimal one-to-one core
    matches = self._hungarian_core()

    # Step 2: attach clauses that Hungarian left unmatched
    matches = self._add_leftover_matches(matches)

    return self._matches_to_dataframe(matches)

  def match_report(self) -> dict[str, pd.DataFrame]:
    """
    Perform matching and return matched pairs plus clauses that received no match
    """

    matches = self._hungarian_core()
    matches = self._add_leftover_matches(matches)

    return {
      "matches": self._matches_to_dataframe(matches),
      "unmatched_a": self._unmatched_a_to_dataframe(matches),
      "unmatched_b": self._unmatched_b_to_dataframe(matches)
    }

  def _matches_to_dataframe(
      self,
      matches: list[tuple[int, int]],
  ) -> pd.DataFrame:
    """
    Convert selected clause matches to a DataFrame
    """

    rows = []

    for i, j in matches:

      precision, recall, f1 = self.score_table.iat[i, j]

      rows.append({
        "clause_in_a_index": i,
        "clause_in_b_index": j,
        "clause_in_a": self.score_table.index[i],
        "clause_in_b": self.score_table.columns[j],
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
      })

    result = pd.DataFrame(
      rows,
      columns=[
        "clause_in_a_index",
        "clause_in_b_index",
        "clause_in_a",
        "clause_in_b",
        "precision",
        "recall",
        "f1"
      ]
    )

    if not result.empty:
      result = result.sort_values(
        ["clause_in_a_index", "f1", "clause_in_b_index"],
        ascending=[True, False, True]
      ).reset_index(drop=True)
    return result

  def _unmatched_a_to_dataframe(
      self,
      matches: list[tuple[int, int]],
  ) -> pd.DataFrame:
    """
    Convert A-side clauses with zero selected matches to a DataFrame
    """

    matched_a = {
      i
      for i, _ in matches
    }

    rows = []

    for i in range(self.n_a):
      if i in matched_a:
        continue

      rows.append({
        "clause_in_a_index": i,
        "clause_in_a": self.score_table.index[i]
      })

    return pd.DataFrame(
      rows,
      columns=[
        "clause_in_a_index",
        "clause_in_a"
      ]
    )

  def _unmatched_b_to_dataframe(
      self,
      matches: list[tuple[int, int]],
  ) -> pd.DataFrame:
    """
    Convert B-side clauses with zero selected matches to a DataFrame
    """

    matched_b = {
      j
      for _, j in matches
    }

    rows = []

    for j in range(self.n_b):
      if j in matched_b:
        continue

      rows.append({
        "clause_in_b_index": j,
        "clause_in_b": self.score_table.columns[j]
      })

    return pd.DataFrame(
      rows,
      columns=[
        "clause_in_b_index",
        "clause_in_b"
      ]
    )
