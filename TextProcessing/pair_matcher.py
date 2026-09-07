import numpy as np
import pandas as pd
from pathlib import Path

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

  tfidf_score_table:
    Optional DataFrame returned by TFIDFScore.create_tfidf_score_table()
    Each cell must contain a TF-IDF cosine similarity score.
    If provided, matching uses the average of the selected BERTScore metric
    and the TF-IDF score.

  use_position_bias:
    If True, matching blends semantic/lexical similarity with normalized
    clause position similarity.

  position_weight:
    Weight assigned to position similarity when use_position_bias is True.
    The semantic/lexical score receives the remaining weight.

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
      tfidf_score_table: pd.DataFrame | None = None,
      min_score: float = 0.0,
      metric: str = "f1",
      capacity: int = 2,
      use_position_bias: bool = False,
      position_weight: float = 0.15,
  ) -> None:
    if metric not in self.METRIC_INDEX:
      raise ValueError(
        "metric must be `precision`, `recall` or `f1`"
      )

    if capacity < 1:
      raise ValueError("capacity must be at least 1.")

    if not 0.0 <= position_weight <= 1.0:
      raise ValueError("position_weight must be between 0.0 and 1.0")

    self.score_table = score_table
    self.tfidf_score_table = tfidf_score_table
    self.min_score = min_score
    self.metric = metric
    self.capacity = capacity
    self.use_position_bias = use_position_bias
    self.position_weight = position_weight

    self.n_a = len(score_table.index)   # Number of clauses in A
    self.n_b = len(score_table.columns) # Number of clauses in B

    self._validate_tfidf_score_table()
    self._scores = self._extract_scores()

  def _validate_tfidf_score_table(self) -> None:
    if self.tfidf_score_table is None:
      return

    if self.tfidf_score_table.shape != self.score_table.shape:
      raise ValueError(
        "tfidf_score_table must have the same shape as score_table"
      )

    if not self.tfidf_score_table.index.equals(self.score_table.index):
      raise ValueError(
        "tfidf_score_table must have the same index as score_table"
      )

    if not self.tfidf_score_table.columns.equals(self.score_table.columns):
      raise ValueError(
        "tfidf_score_table must have the same columns as score_table"
      )

  def _extract_scores(self) -> np.ndarray:
    """
    Build the numeric alignment score matrix used for matching.
    """
    scores = np.zeros(
      (self.n_a, self.n_b),
      dtype=float
    )

    for i in range(self.n_a):
      for j in range(self.n_b):
        scores[i, j] = self._alignment_score_at(i, j)

    return scores

  def _bert_score_at(
      self,
      i: int,
      j: int,
  ) -> float:
    metric_index = self.METRIC_INDEX[self.metric]
    cell = self.score_table.iat[i, j]
    return float(cell[metric_index])

  def _tfidf_score_at(
      self,
      i: int,
      j: int,
  ) -> float | None:
    if self.tfidf_score_table is None:
      return None

    return float(self.tfidf_score_table.iat[i, j])

  def _semantic_lexical_score_at(
      self,
      i: int,
      j: int,
  ) -> float:
    bert_score = self._bert_score_at(i, j)
    tfidf_score = self._tfidf_score_at(i, j)

    if tfidf_score is None:
      return bert_score

    return (bert_score + tfidf_score) / 2.0

  def _position_score_at(
      self,
      i: int,
      j: int,
  ) -> float:
    position_a = i / (self.n_a - 1) if self.n_a > 1 else 0.0
    position_b = j / (self.n_b - 1) if self.n_b > 1 else 0.0
    return 1.0 - abs(position_a - position_b)

  def _alignment_score_at(
      self,
      i: int,
      j: int,
  ) -> float:
    semantic_lexical_score = self._semantic_lexical_score_at(i, j)

    if not self.use_position_bias:
      return semantic_lexical_score

    position_score = self._position_score_at(i, j)
    semantic_weight = 1.0 - self.position_weight

    return (
      semantic_weight * semantic_lexical_score
      + self.position_weight * position_score
    )

  def alignment_score_table(self) -> pd.DataFrame:
    return pd.DataFrame(
      self._scores,
      index=self.score_table.index,
      columns=self.score_table.columns
    )

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

  def append_matches_to_markdown(
      self,
      output_path: str | Path,
      filename: str,
      clause_pairs: pd.DataFrame,
  ) -> None:
    output_path = Path(output_path)

    with output_path.open("a", encoding="utf-8") as file:
      file.write(f"## Filename: {filename}\n\n")
      file.write(f"Metric: {self.metric}\n\n")
      file.write(f"Minimum score: {self.min_score}\n\n")
      file.write(f"Capacity: {self.capacity}\n\n")

      if clause_pairs.empty:
        file.write("No clause pairs matched.\n\n")
        return

      try:
        file.write(clause_pairs.to_markdown(index=False))
        file.write("\n\n")
      except ImportError:
        file.write("```text\n")
        file.write(clause_pairs.to_string(index=False))
        file.write("\n```\n\n")

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

      row = {
        "clause_in_a_index": i,
        "clause_in_b_index": j,
        "clause_in_a": self.score_table.index[i],
        "clause_in_b": self.score_table.columns[j],
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
      }

      if self.tfidf_score_table is not None:
        row["tfidf"] = float(self.tfidf_score_table.iat[i, j])

      if self.use_position_bias:
        row["semantic_lexical_score"] = self._semantic_lexical_score_at(i, j)
        row["position_score"] = self._position_score_at(i, j)

      row["match_score"] = float(self._scores[i, j])
      rows.append(row)

    columns = [
      "clause_in_a_index",
      "clause_in_b_index",
      "clause_in_a",
      "clause_in_b",
      "precision",
      "recall",
      "f1"
    ]

    if self.tfidf_score_table is not None:
      columns.append("tfidf")

    if self.use_position_bias:
      columns.extend([
        "semantic_lexical_score",
        "position_score"
      ])

    columns.append("match_score")

    result = pd.DataFrame(
      rows,
      columns=columns
    )

    if not result.empty:
      result = result.sort_values(
        ["clause_in_a_index", "match_score", "clause_in_b_index"],
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
