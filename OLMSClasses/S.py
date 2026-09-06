# This script calculates a structural similarity score based on syntactic similarity
from collections import Counter

import pandas as pd

from TextProcessing.deprel import DependencyToken

class S:
  """
  Structural similarity over semantically matched clause pairs.

  Implements:

    Struct(D_a, D_b) = (1 / |R|) * sum J(P(c_i), P(c_j))

  where R is the set of matched clause pairs returned by PairMatcher.match(),
  P(c) is a multiset of syntactic patterns from clause c, and J is multiset
  Jaccard similarity.
  """

  REQUIRED_COLUMNS = {
    "clause_in_a_index",
    "clause_in_b_index",
  }

  def __init__(
      self,
      match_table: pd.DataFrame,
      clauses_a_tokens: list[list[DependencyToken]],
      clauses_b_tokens: list[list[DependencyToken]],
      skip_punct: bool = True,
  ) -> None:
    missing_columns = self.REQUIRED_COLUMNS - set(match_table.columns)
    if missing_columns:
      raise ValueError(
        f"The paired clauses table from pair_matcher.py is missing required columns: {sorted(missing_columns)}"
      )

    self.match_table = match_table.copy()
    self.clauses_a_tokens = clauses_a_tokens
    self.clauses_b_tokens = clauses_b_tokens
    self.skip_punct = skip_punct

    self._validate_clause_indices()
    return

  def structure_score(self) -> float:
    """
    Return the average multiset Jaccard similarity over matched clause pairs.
    If there are no matched pairs, return 0.0.
    """
    if self.match_table.empty:
      return 0.0

    scores = self.pair_scores()
    return float(scores["structurality"].mean())

  def pair_scores(self) -> pd.DataFrame:
    """
    Return structurality diagnostics for each matched clause pair.
    """
    rows = []

    for _, row in self.match_table.iterrows():
      clause_in_a_index = int(row["clause_in_a_index"])
      clause_in_b_index = int(row["clause_in_b_index"])

      patterns_a = self._syntactic_patterns(
        self.clauses_a_tokens[clause_in_a_index]
      )
      patterns_b = self._syntactic_patterns(
        self.clauses_b_tokens[clause_in_b_index]
      )

      rows.append({
        "clause_in_a_index": clause_in_a_index,
        "clause_in_b_index": clause_in_b_index,
        "structurality": self._multiset_jaccard(patterns_a, patterns_b),
        "n_patterns_a": int(patterns_a.total()),
        "n_patterns_b": int(patterns_b.total()),
        "n_pattern_types_a": len(patterns_a),
        "n_pattern_types_b": len(patterns_b),
        "n_shared_pattern_types": len(set(patterns_a) & set(patterns_b)),
      })

    return pd.DataFrame(
      rows,
      columns=[
        "clause_in_a_index",
        "clause_in_b_index",
        "structurality",
        "n_patterns_a",
        "n_patterns_b",
        "n_pattern_types_a",
        "n_pattern_types_b",
        "n_shared_pattern_types",
      ]
    )

  def _syntactic_patterns(
      self,
      clause_tokens: list[DependencyToken],
  ) -> Counter[tuple[str, str, str]]:
    """
    Build the multiset of <pos_t, dep_r, pos_h> patterns for one clause.
    Only include dependency edges where both token and head are inside the same
    clause.
    """
    token_by_id = {
      token.token_id: token
      for token in clause_tokens
    }

    patterns: Counter[tuple[str, str, str]] = Counter()

    for token in clause_tokens:
      if self.skip_punct and token.upos == "PUNCT":
        continue

      if token.head_token_id is None:
        continue

      head = token_by_id.get(token.head_token_id)
      if head is None:
        continue

      if self.skip_punct and head.upos == "PUNCT":
        continue

      patterns[
        (
          token.upos,
          token.dependency_relation,
          head.upos,
        )
      ] += 1

    return patterns

  def _multiset_jaccard(
      self,
      patterns_a: Counter[tuple[str, str, str]],
      patterns_b: Counter[tuple[str, str, str]],
  ) -> float:
    keys = set(patterns_a) | set(patterns_b)

    if not keys:
      return 0.0

    numerator = sum(
      min(patterns_a[key], patterns_b[key])
      for key in keys
    )

    denominator = sum(
      max(patterns_a[key], patterns_b[key])
      for key in keys
    )

    if denominator == 0:
      return 0.0

    return numerator / denominator

  def _validate_clause_indices(self) -> None:
    if self.match_table.empty:
      return

    max_a = len(self.clauses_a_tokens) - 1
    max_b = len(self.clauses_b_tokens) - 1

    bad_a = sorted(
      {
        int(index)
        for index in self.match_table["clause_in_a_index"]
        if int(index) < 0 or int(index) > max_a
      }
    )

    bad_b = sorted(
      {
        int(index)
        for index in self.match_table["clause_in_b_index"]
        if int(index) < 0 or int(index) > max_b
      }
    )

    errors = []
    if bad_a:
      errors.append(
        f"A-side clause indices out of range 0..{max_a}: {bad_a}"
      )

    if bad_b:
      errors.append(
        f"B-side clause indices out of range 0..{max_b}: {bad_b}"
      )

    if errors:
      raise ValueError("; ".join(errors))
