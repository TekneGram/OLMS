import pandas as pd
from collections.abc import Sequence

from TextProcessing.shared_bert_scorer import get_bert_scorer

class BertScore:
  def __init__(
      self,
      text_a: list[str],
      text_b: list[str]
  ) -> None:
    self.text_a = text_a
    self.text_b = text_b
    self.scorer = get_bert_scorer()

  def create_bert_score_table(self) -> pd.DataFrame:
    return self.create_bert_score_tables([(self.text_a, self.text_b)], self.scorer)[0]

  @staticmethod
  def create_bert_score_tables(
      text_pairs: Sequence[tuple[list[str], list[str]]],
      scorer=None,
      context_cache=None,
  ) -> list[pd.DataFrame]:
    """Score complete sentence matrices together and reconstruct each intact table."""
    if not text_pairs:
      return []

    if scorer is None and context_cache is None:
      scorer = get_bert_scorer()
    candidates, references = [], []
    boundaries = []
    for text_a, text_b in text_pairs:
      if not text_a or not text_b:
        raise ValueError("BERTScore sentence matrices require non-empty sentence lists.")
      start = len(candidates)
      candidates.extend(text_a_i for text_a_i in text_a for _ in text_b)
      references.extend(text_b_i for _ in text_a for text_b_i in text_b)
      boundaries.append((start, len(text_a) * len(text_b), text_a, text_b))

    score_source = context_cache if context_cache is not None else scorer
    precision, recall, f1 = score_source.score(candidates, references)
    if not (len(precision) == len(recall) == len(f1) == len(candidates)):
      raise ValueError("BERTScore returned an unexpected number of sentence scores.")

    tables = []
    for start, cell_count, text_a, text_b in boundaries:
      end = start + cell_count
      if end > len(candidates):
        raise ValueError("BERTScore sentence matrix boundary exceeds the batched result.")
      values = [
        [
          [
            precision[start + i * len(text_b) + j].item(),
            recall[start + i * len(text_b) + j].item(),
            f1[start + i * len(text_b) + j].item(),
          ]
          for j in range(len(text_b))
        ]
        for i in range(len(text_a))
      ]
      table = pd.DataFrame(values, index=text_a, columns=text_b)
      if table.shape != (len(text_a), len(text_b)):
        raise ValueError("BERTScore did not reconstruct a complete sentence matrix.")
      tables.append(table)
    return tables

  def create_bert_score_table_sliding_window(
      self,
      window_size: int = 3,
  ) -> pd.DataFrame:
    if window_size < 1:
      raise ValueError("window_size must be at least 1.")

    if not self.text_a or not self.text_b:
      return pd.DataFrame(index=self.text_a, columns=self.text_b)

    windows_a = self._sliding_windows(self.text_a, window_size)
    windows_b = self._sliding_windows(self.text_b, window_size)

    pairs = [(x, y) for x in windows_a for y in windows_b]
    candidates = [x for x, y in pairs]
    references = [y for x, y in pairs]

    precision, recall, f1 = self.scorer.score(candidates, references)

    precision = precision.reshape(len(self.text_a), len(self.text_b))
    recall = recall.reshape(len(self.text_a), len(self.text_b))
    f1 = f1.reshape(len(self.text_a), len(self.text_b))

    table = pd.DataFrame(
      [
        [
          [
            precision[i, j].item(),
            recall[i, j].item(),
            f1[i, j].item(),
          ]
          for j in range(len(self.text_b))
        ]
        for i in range(len(self.text_a))
      ],
      index=self.text_a,
      columns=self.text_b
    )
    return table

  def _sliding_windows(
      self,
      texts: list[str],
      window_size: int,
  ) -> list[str]:
    return [
      " ".join(
        texts[min(start + offset, len(texts) - 1)]
        for offset in range(window_size)
      )
      for start in range(len(texts))
    ]
