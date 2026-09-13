from pathlib import Path
from typing import Any

from SLM.embedding import EmbeddingModel
from TextProcessing.deprel import DependencyParse


ResponseRow = dict[str, Any]
PreparedResponse = tuple[DependencyParse, list[str]]
VectorScores = dict[str, float]


class OLMSPairScorer:
  """Score saved response pairs while caching parsed responses per essay."""

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
    self.active_essay: str | None = None

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

    if first["essay_file"] != self.active_essay:
      self.cache.clear()
      self.active_essay = first["essay_file"]

    parsed_a, clauses_a = self._prepare(first)
    parsed_b, clauses_b = self._prepare(second)
    score_table = BertScore(clauses_a, clauses_b).create_bert_score_table()
    vector = OLMSVector(
      score_table, parsed_a, parsed_b, first["response"], second["response"], self.embedding_model)
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
