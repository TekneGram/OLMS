from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

import configuration
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
    return self._score_pair_batch([(first, second)])[0]

  def score_pairs(
      self,
      pairs: Sequence[tuple[ResponseRow, ResponseRow]],
  ) -> Iterator[VectorScores]:
    """Score complete response pairs in bounded BERTScore batches."""
    batch_size = configuration.bertscore_pair_batch_size
    if not isinstance(batch_size, int) or batch_size < 1:
      raise ValueError("bertscore_pair_batch_size must be a positive integer.")
    for start in range(0, len(pairs), batch_size):
      yield from self._score_pair_batch(pairs[start:start + batch_size])

  def _score_pair_batch(
      self,
      pairs: Sequence[tuple[ResponseRow, ResponseRow]],
  ) -> list[VectorScores]:
    """Score one batch of intact response pairs without carrying partial matrices."""
    from OLMSClasses.M import M
    from OLMSClasses.OLMS_vector import OLMSVector
    from TextProcessing.bertscorer import BertScore

    if not pairs:
      return []
    essay_file = pairs[0][0]["essay_file"]
    if any(first["essay_file"] != essay_file or second["essay_file"] != essay_file
           for first, second in pairs):
      raise ValueError("BERTScore pair batches must contain one essay.")
    self._activate_essay(essay_file)

    prepared = [(self._prepare(first), self._prepare(second)) for first, second in pairs]
    score_tables = BertScore.create_bert_score_tables([
      (clauses_a, clauses_b) for (_, clauses_a), (_, clauses_b) in prepared
    ])
    meaning_scores = M.batch_meaning_similarity_bertscores([
      (first["response"], second["response"]) for first, second in pairs
    ])
    if not (len(score_tables) == len(meaning_scores) == len(pairs)):
      raise ValueError("BERTScore batch did not return every complete response pair.")

    results = []
    for (first, second), ((parsed_a, _), (parsed_b, _)), score_table, meaning_score in zip(
        pairs, prepared, score_tables, meaning_scores, strict=True):
      vector = OLMSVector(
        score_table, parsed_a, parsed_b, first["response"], second["response"], self.embedding_model,
        embedding_a=self.embedding_cache.get(self._response_key(first)),
        embedding_b=self.embedding_cache.get(self._response_key(second)),
        meaning_bertscore=meaning_score,
      )
      results.append(vector.create_olms_vector())

      if self.diagnostics_dir is not None:
        label = (f"{self.active_essay}: {first['prompt']}{first['response_index']} / "
                 f"{second['prompt']}{second['response_index']}")
        vector.structure.append_structure_to_markdown(self.diagnostics_dir / "structure.md", label)
    return results

  def __call__(
      self,
      first: ResponseRow,
      second: ResponseRow,
  ) -> VectorScores:
    return self.score_pair(first, second)

  def close(self) -> None:
    self.embedding_model.close()
