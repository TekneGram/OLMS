from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

from SLM.embedding import EmbeddingModel
from TextProcessing.deprel import DependencyParse


ResponseRow = dict[str, Any]
PreparedResponse = tuple[DependencyParse, list[str]]
VectorScores = dict[str, float]
ResponseKey = tuple[str, str, int]


class OLMSPairScorer:
  """Score saved response pairs while caching parsed responses per essay."""

  EMBEDDING_BATCH_SIZE = 16

  def __init__(
      self,
      parser_dir: str | Path,
      diagnostics_dir: str | Path | None = None,
  ) -> None:
    from TextProcessing.deprel import DependencyParser

    self.parser = DependencyParser(parser_dir)
    self.embedding_model = EmbeddingModel()
    self.diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir is not None else None
    self.cache: dict[tuple[str, int], PreparedResponse] = {}
    self.embedding_cache: dict[ResponseKey, np.ndarray] = {}
    self.active_essay: str | None = None

  @staticmethod
  def _response_key(row: ResponseRow) -> ResponseKey:
    return (row["essay_file"], row["prompt"], row["response_index"])

  def _activate_essay(self, essay_file: str) -> None:
    """Clear per-essay caches when scoring advances to another essay."""
    if essay_file != self.active_essay:
      self.cache.clear()
      self.embedding_cache.clear()
      self.active_essay = essay_file

  def prepare_embeddings(self, responses: Sequence[ResponseRow]) -> None:
    """Encode each unique response needed for the active essay once."""
    if not responses:
      return

    essay_file = responses[0]["essay_file"]
    if any(row["essay_file"] != essay_file for row in responses):
      raise ValueError("Embedding preparation requires responses from one essay.")
    self._activate_essay(essay_file)

    unique_responses: dict[ResponseKey, ResponseRow] = {}
    for row in responses:
      unique_responses.setdefault(self._response_key(row), row)

    missing = [
      row for key, row in unique_responses.items()
      if key not in self.embedding_cache
    ]
    for start in range(0, len(missing), self.EMBEDDING_BATCH_SIZE):
      batch = missing[start:start + self.EMBEDDING_BATCH_SIZE]
      embeddings = self.embedding_model.encode([row["response"] for row in batch])
      for row, embedding in zip(batch, embeddings, strict=True):
        self.embedding_cache[self._response_key(row)] = embedding

  def _prepare(
      self,
      row: ResponseRow,
  ) -> PreparedResponse:
    from TextProcessing.sentence_splitter import SentenceSplitter

    key = (row["prompt"], row["response_index"])
    if key not in self.cache:
      name = f"{row['essay_file']}_{key[0]}_{key[1]}"
      parsed = self.parser.parse(name, row["response"])
      splitter = SentenceSplitter(parsed)
      clauses = [splitter.token_text(tokens) for tokens in splitter.split_all_tokens()]
      if not clauses:
        raise ValueError(f"No sentences could be parsed for {name}.")
      self.cache[key] = (parsed, clauses)
      if self.diagnostics_dir is not None:
        SentenceSplitter.append_clause_sets_to_markdown(
          self.diagnostics_dir / "clauses.md", name, {"Sentences": clauses})
    return self.cache[key]

  def score_pair(
      self,
      first: ResponseRow,
      second: ResponseRow,
  ) -> VectorScores:
    from OLMSClasses.OLMS_vector import OLMSVector
    from TextProcessing.bertscorer import BertScore

    self._activate_essay(first["essay_file"])

    parsed_a, clauses_a = self._prepare(first)
    parsed_b, clauses_b = self._prepare(second)
    score_table = BertScore(clauses_a, clauses_b).create_bert_score_table()
    vector = OLMSVector(
      score_table, parsed_a, parsed_b, first["response"], second["response"], self.embedding_model,
      embedding_a=self.embedding_cache.get(self._response_key(first)),
      embedding_b=self.embedding_cache.get(self._response_key(second)),
    )
    result = vector.create_olms_vector()

    if self.diagnostics_dir is not None:
      label = (f"{self.active_essay}: {first['prompt']}{first['response_index']} / "
               f"{second['prompt']}{second['response_index']}")
      vector.structure.append_structure_to_markdown(self.diagnostics_dir / "structure.md", label)

    return result

  def __call__(
      self,
      first: ResponseRow,
      second: ResponseRow,
  ) -> VectorScores:
    return self.score_pair(first, second)

  def close(self) -> None:
    self.embedding_model.close()
