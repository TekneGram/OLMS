from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
import pandas as pd


class OLMSVectorTable:
  """Validate complete per-essay pair tables and expose bootstrap lookups."""

  COMPONENTS = ["org", "lex", "meaning", "struct"]
  KEYS = ["essay_file", "prompt_1", "response_index_1", "prompt_2", "response_index_2"]
  COLUMNS = KEYS + ["comparison_type", "response_count"] + COMPONENTS

  def __init__(self, source):
    self.frame = (pd.read_csv(Path(source), keep_default_na=False)
                  if isinstance(source, (str, Path)) else source.copy())
    self.validate_rows(self.frame)
    if self.frame.empty:
      raise ValueError("The vector table is empty.")
    self._validate_coverage()

  @classmethod
  def validate_rows(cls, frame):
    missing = set(cls.COLUMNS) - set(frame.columns)
    if missing:
      raise ValueError(f"Missing vector columns {sorted(missing)}. Regenerate legacy vectors with the vectors command.")
    if frame[cls.COLUMNS].isna().any().any():
      raise ValueError("Vector rows contain missing values.")
    if frame.duplicated(cls.KEYS).any():
      raise ValueError("Duplicate response pairs in vector table.")
    if not frame.essay_file.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
      raise ValueError("Every vector requires an essay filename.")

    for column in ["response_index_1", "response_index_2", "response_count"]:
      values = pd.to_numeric(frame[column], errors="raise")
      if not np.isfinite(values).all() or (values < 1).any() or (values % 1 != 0).any():
        raise ValueError(f"{column} must contain positive integers.")
      frame[column] = values.astype(int)

    if frame.duplicated(cls.KEYS).any():
        raise ValueError("Duplicate response pairs in vector table.")

    frame[cls.COMPONENTS] = frame[cls.COMPONENTS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[cls.COMPONENTS].to_numpy()).all():
      raise ValueError("Non-finite OLMS scores in vector table.")

    valid_prompts = frame.prompt_1.isin(["A", "B"]) & frame.prompt_2.isin(["A", "B"])
    valid_order = ((frame.prompt_1 == frame.prompt_2) &
                    (frame.response_index_1 <= frame.response_index_2)) | (
                      (frame.prompt_1 == "A") & (frame.prompt_2 == "B"))
    valid_type = frame.comparison_type == frame.prompt_1 + frame.prompt_2

    if not (valid_prompts & valid_order & valid_type).all():
      raise ValueError("Expected canonical AA, BB, or AB response pairs.")

    if (frame.response_count < 2).any() or any(
      (frame[column] > frame.response_count).any()
      for column in ["response_index_1", "response_index_2"]
    ):
      raise ValueError("Response indices must be within 1..response_count, with at least two responses.")

  def _validate_coverage(self):
    if self.frame.response_count.nunique() != 1:
      raise ValueError("All essay-prompt conditions must use the same response count.")

    for essay, group in self.frame.groupby("essay_file", sort=True):
      count = int(group.response_count.iloc[0])
      ids = range(1, count + 1)
      within = set(combinations_with_replacement(ids, 2))

      for comparison in ["AA", "BB", "AB"]:
        rows = group[group.comparison_type == comparison]
        actual = set(zip(rows.response_index_1, rows.response_index_2))
        expected = set(product(ids, ids)) if comparison == "AB" else within

        if actual != expected:
          raise ValueError(f"Incomplete {comparison} vectors for {essay}: expected {len(expected)}, found {len(actual)}. Include self comparisons for bootstrapping.")

  @property
  def essays(self):
    return sorted(self.frame.essay_file.unique())

  @property
  def response_count(self):
    return int(self.frame.response_count.iloc[0])

  def select(self, essay, comparison, include_self=False):
    rows = self.frame[(self.frame.essay_file == essay) & (self.frame.comparison_type == comparison)]
    if not include_self and comparison != "AB":
      rows = rows[rows.response_index_1 != rows.response_index_2]

    return rows.copy()

  def matrices(self, essay):
    result = {}
    for comparison in ["AA", "BB", "AB"]:
      matrix = np.empty((self.response_count, self.response_count, 4), dtype=float)
      rows = self.select(essay, comparison, include_self=True)
      i = rows.response_index_1.to_numpy() - 1
      j = rows.response_index_2.to_numpy() - 1
      values = rows[self.COMPONENTS].to_numpy(dtype=float)
      matrix[i, j] = values
      if comparison != "AB":
        matrix[j, i] = values
      result[comparison] = matrix

    return result